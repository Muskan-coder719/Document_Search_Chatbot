import os
import io
import json
import pandas as pd
from PyPDF2 import PdfReader
from pptx import Presentation
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.discovery import build
from google.oauth2 import service_account

# ✅ Authenticate Google Drive using env variable
def authenticate_drive():
    try:
        service_account_info = json.loads(os.environ.get("GCP_SERVICE_ACCOUNT"))
        creds = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        service = build('drive', 'v3', credentials=creds)
        return service
    except Exception as e:
        raise RuntimeError(f"Google Drive Authentication failed: {e}")

# ✅ Extract text from PPTX
def extract_text_from_pptx(local_path):
    try:
        prs = Presentation(local_path)
        slide_texts = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    slide_texts.append(shape.text)
        return '\n'.join(slide_texts)
    except Exception as e:
        print(f"⚠️ Could not extract PPTX text from {local_path}: {e}")
        return ""

# ✅ Fetch documents from Google Drive
def fetch_documents():
    service = authenticate_drive()
    docs, sources, file_ids, file_paths = [], [], [], []
    os.makedirs('downloaded_files', exist_ok=True)

    def process_file(file, folder_name):
        file_id = file['id']
        file_name = file['name']
        mime_type = file['mimeType']
        source_label = f"{folder_name} / {file_name}"
        text = ""
        local_path = None

        try:
            if mime_type == 'application/pdf':
                request = service.files().get_media(fileId=file_id)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                while not downloader.next_chunk()[1]:
                    pass
                fh.seek(0)
                local_path = os.path.join('downloaded_files', file_name)
                with open(local_path, 'wb') as f:
                    f.write(fh.read())
                fh.seek(0)
                reader = PdfReader(fh)
                for page in reader.pages:
                    text += page.extract_text() or ""

            elif mime_type == 'application/vnd.google-apps.document':
                request = service.files().export(fileId=file_id, mimeType='text/plain')
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                while not downloader.next_chunk()[1]:
                    pass
                fh.seek(0)
                text = fh.read().decode('utf-8')

            elif mime_type in [
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            ]:
                request = service.files().get_media(fileId=file_id)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                while not downloader.next_chunk()[1]:
                    pass
                fh.seek(0)
                local_path = os.path.join('downloaded_files', file_name)
                with open(local_path, 'wb') as f:
                    f.write(fh.read())

                if mime_type.endswith('spreadsheetml.sheet'):
                    try:
                        xls = pd.read_excel(local_path, sheet_name=None)
                        text = '\n'.join(
                            ' '.join(df.fillna('').astype(str).values.flatten())
                            for df in xls.values()
                        )
                    except Exception as e:
                        print(f"⚠️ Excel read error: {file_name} — {e}")
                elif mime_type.endswith('presentationml.presentation'):
                    text = extract_text_from_pptx(local_path)

            else:
                print(f"⚠️ Skipping unsupported type: {mime_type} ({file_name})")
                return

            if text.strip() or local_path:
                docs.append(text)
                sources.append(source_label)
                file_ids.append(file_id)
                file_paths.append(local_path)

        except Exception as e:
            print(f"❌ Error processing {source_label}: {e}")

    # ✅ Process root files
    root_files = service.files().list(
        q="'root' in parents and trashed = false",
        fields="files(id, name, mimeType)"
    ).execute().get('files', [])
    for file in root_files:
        process_file(file, "Root")

    # ✅ Process folders
    folders = service.files().list(
        q="mimeType = 'application/vnd.google-apps.folder' and trashed = false",
        fields="files(id, name)"
    ).execute().get('files', [])
    for folder in folders:
        folder_id = folder['id']
        folder_name = folder['name']
        files = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="files(id, name, mimeType)"
        ).execute().get('files', [])
        for file in files:
            process_file(file, folder_name)

    return docs, sources, file_ids, file_paths
