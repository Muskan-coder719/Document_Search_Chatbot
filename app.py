import os
import json
import streamlit as st
from datetime import datetime
from sentence_transformers import SentenceTransformer
import pickle
from embeddings_utils import load_embeddings
from faiss_utils import create_faiss_index
from google.oauth2 import service_account
from googleapiclient.discovery import build

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="FamilyTLC Search Bot", layout="wide", page_icon="🧠")
st.markdown("""
<div style="
    background: linear-gradient(90deg, #004d99, #007acc);
    padding: 2rem;
    border-radius: 12px;
    text-align: center;
    color: white;
    font-size: 2.5rem;
    font-weight: bold;
    margin-bottom: 2rem;">
    🧠 FamilyTLC Document Search Bot
</div>""", unsafe_allow_html=True)

# ---------------- CSS ----------------
st.markdown("""
<style>
.metric-card {padding:1.1em 1.2em;border-radius:12px;margin-bottom:10px;color:white;text-align:center;font-weight:bold;font-size:1.15em;box-shadow:0 3px 14px rgba(75,75,75,0.03);}
.gradient-blue {background:linear-gradient(90deg,#2834d9,#47c6ff);}
.gradient-magenta {background:linear-gradient(90deg,#d138d1,#f98fff);}
.gradient-yellow {background:linear-gradient(90deg,#f7e96c,#efb12d);}
.gradient-green {background:linear-gradient(90deg,#5de684,#14bca9);}
.stat-label {font-size:0.98em;opacity:0.94;}
.stat-number {font-size:2em;margin-bottom:4px;}
.tag-btn {display:inline-block;padding:8px 17px;margin:4px 4px 6px 0;background:#ecf7fe;border:none;border-radius:6px;color:#1c3d6f;font-size:1em;font-weight:500;cursor:pointer;transition:0.13s;}
.tag-btn:hover {background:#3567be;color:white;}
.result-card {background:#fff;border-radius:1em;padding:1.3em;margin:1em 0 1.5em 0;box-shadow:0 2px 12px #edf1f9;border-left:5px solid #2196f3;}
.status-success {background:#e0fbe7;color:#21794a;padding:0.15em 0.7em;border-radius:7px;font-size:1em;}
.status-warning {background:#fffbe3;color:#a26c06;padding:0.17em 0.7em;border-radius:7px;font-size:1em;}
.status-error {background:#fff0f0;color:#ad3939;padding:0.13em 0.7em;border-radius:7px;font-size:1em;}
.stTabs [role="tablist"] {border-bottom:none}
input[type="text"], textarea, .stTextInput>div>div>input {border-radius:8px !important;}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
try:
    embedding_matrix = load_embeddings('data/embeddings.npy')
    with open('data/sources.pkl', 'rb') as f:
        sources = pickle.load(f)
    with open('data/text_chunks.pkl', 'rb') as f:
        text_chunks = pickle.load(f)
    with open('data/file_ids.pkl', 'rb') as f:
        file_ids = pickle.load(f)
    DOCUMENT_COUNT = len(set(file_ids))
except Exception:
    sources, text_chunks, file_ids, embedding_matrix = [], [], [], None
    DOCUMENT_COUNT = 0

SEARCHES_TODAY = 42
AVG_RESPONSE = "1.2s"
SUCCESS_RATE = "98.9%"
USERS_TODAY = 14
POPULAR_SEARCHES = ["Vacation Policy", "Work From Home", "Sick Leave", "Team Meetings"]

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("🛠 Admin Tools")
    st.markdown("### 📊 Quick Stats")
    st.markdown(f"""
    <div class='metric-card gradient-blue'><div class='stat-number'>{DOCUMENT_COUNT}</div><div class='stat-label'>Documents Indexed</div></div>
    <div class='metric-card gradient-magenta'><div class='stat-number'>{USERS_TODAY}</div><div class='stat-label'>Active Users Today</div></div>
    <div class='metric-card gradient-green'><div class='stat-number'>{SUCCESS_RATE}</div><div class='stat-label'>Success Rate</div></div>
    """, unsafe_allow_html=True)

# ---------------- MAIN METRICS ----------------
cols = st.columns(4)
for idx, (icon, val, label, color) in enumerate([
    ("📄", DOCUMENT_COUNT, "Documents", "gradient-blue"),
    ("🔍", SEARCHES_TODAY, "Searches Today", "gradient-magenta"),
    ("⚡", AVG_RESPONSE, "Avg Response", "gradient-yellow"),
    ("✅", SUCCESS_RATE, "Success Rate", "gradient-green"),
]):
    with cols[idx]:
        st.markdown(f"""
        <div class="metric-card {color}">
            <div style="font-size:2em;">{icon}</div>
            <div class="stat-number">{val}</div>
            {label}
        </div>
        """, unsafe_allow_html=True)

# ---------------- POPULAR SEARCH TAGS ----------------
st.markdown("## <b>Quick Start - Popular Searches</b>", unsafe_allow_html=True)
tag_cols = st.columns(min(len(POPULAR_SEARCHES), 3))
for i, tag in enumerate(POPULAR_SEARCHES):
    tag_cols[i % len(tag_cols)].markdown(f"<button class='tag-btn'>{tag}</button>", unsafe_allow_html=True)

# ---------------- TABS ----------------
tab_search, tab_faq, tab_about = st.tabs(["🔍 Search", "📖 FAQ", "ℹ About"])

# ---------------- MODEL ----------------
model = None
index = None
if embedding_matrix is not None and len(embedding_matrix):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = create_faiss_index(embedding_matrix)

# ✅ Google Drive Auth (using environment variable)
def get_drive_service():
    try:
        service_account_info = json.loads(os.environ["GCP_SERVICE_ACCOUNT"])
        creds = service_account.Credentials.from_service_account_info(service_account_info, scopes=["https://www.googleapis.com/auth/drive.metadata.readonly"])
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        st.error(f"Google Drive authentication error: {e}")
        return None

# ---------------- SEARCH TAB ----------------
with tab_search:
    with st.form(key="search_form"):
        user_email = st.text_input("Enter your Google email:", key="email", placeholder="you@familytlc.com")
        question = st.text_input("What would you like to know today?", key="query", placeholder="e.g., Show me sick leave policy")
        submitted = st.form_submit_button("Search Documents")

    if submitted:
        if not user_email:
            st.markdown('<div class="status-error">Please enter your email address.</div>', unsafe_allow_html=True)
        elif not question:
            st.markdown('<div class="status-warning">Please enter a question.</div>', unsafe_allow_html=True)
        elif model is None or index is None:
            st.markdown('<div class="status-error">Search not initialized. Contact admin.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status-success">Searching for: <b>{question}</b></div>', unsafe_allow_html=True)
            drive_service = get_drive_service()
            allowed_file_ids = set()
            if drive_service:
                try:
                    query = f"'{user_email}' in readers"
                    response = drive_service.files().list(q=query, fields="files(id)").execute()
                    allowed_file_ids = {f['id'] for f in response.get('files', [])}
                except Exception as e:
                    st.error(f"Drive API error: {e}")

            query_embedding = model.encode([question]).astype("float32")
            distances, indices = index.search(query_embedding, 10)
            results = [(d, idx) for d, idx in zip(distances[0], indices[0])]

            if not results:
                st.warning("No results found.")
            else:
                for _, idx in results[:3]:
                    file_name = sources[idx]
                    snippet = text_chunks[idx][:500].replace("\n", " ")
                    file_id = file_ids[idx]
                    link = f"https://drive.google.com/file/d/{file_id}/view"
                    if file_id in allowed_file_ids:
                        st.markdown(f"""
                        <div class="result-card">
                            <h4>📄 {file_name}</h4>
                            <p>{snippet}...</p>
                            <a href="{link}" target="_blank">🔗 View Document</a>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="status-error">🔒 No access to <b>{file_name}</b></div>', unsafe_allow_html=True)

# ---------------- FAQ TAB ----------------
with tab_faq:
    st.markdown("""
    <div class="result-card">
        <p><b>Q:</b> What can I search?<br>A: Any internal file, handbook, or HR doc you have permission to access on Google Drive.</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- ABOUT TAB ----------------
with tab_about:
    st.markdown("""
    <div class="result-card">
        <b>FamilyTLC Search Bot</b><br>
        - 🔎 Semantic search using Sentence Transformers + FAISS<br>
        - 🔒 Permission-aware search with Google Drive API
    </div>
    """, unsafe_allow_html=True)
