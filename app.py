import streamlit as st
from datetime import datetime
from sentence_transformers import SentenceTransformer
import pickle
import subprocess
import os
import json
from embeddings_utils import load_embeddings
from faiss_utils import create_faiss_index
from google.oauth2 import service_account
from googleapiclient.discovery import build

# ------------- PAGE CONFIG / HEADER -------------
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

# ------------- CSS STYLE -------------
st.markdown("""
<style>
.metric-card {
    padding:1.1em 1.2em; border-radius:12px; margin-bottom:10px; color:white;
    text-align:center; font-weight:bold; font-size:1.15em; box-shadow:0 3px 14px rgba(75,75,75,0.03);
}
.gradient-blue {background:linear-gradient(90deg,#2834d9,#47c6ff);}
.gradient-magenta {background:linear-gradient(90deg,#d138d1,#f98fff);}
.gradient-yellow {background:linear-gradient(90deg,#f7e96c,#efb12d);}
.gradient-green {background:linear-gradient(90deg,#5de684,#14bca9);}
.stat-label {font-size:0.98em; opacity:0.94;}
.stat-number {font-size:2em; margin-bottom:4px;}
.tag-btn {
  display:inline-block;padding:8px 17px; margin:4px 4px 6px 0;
  background: #ecf7fe; border:none; border-radius:6px;color:#1c3d6f;
  font-size:1em;font-weight:500;cursor:pointer;transition:0.13s;
}
.tag-btn:hover {background:#3567be;color:white;}
.result-card {
    background: #fff; border-radius: 1em; padding: 1.3em; margin: 1em 0 1.5em 0; box-shadow:0 2px 12px #edf1f9;
    border-left:5px solid #2196f3;
}
.status-success {background:#e0fbe7;color:#21794a;padding:0.15em 0.7em;border-radius:7px;font-size:1em;}
.status-warning {background:#fffbe3;color:#a26c06;padding:0.17em 0.7em;border-radius:7px;font-size:1em;}
.status-error {background:#fff0f0;color:#ad3939;padding:0.13em 0.7em;border-radius:7px;font-size:1em;}
</style>
""", unsafe_allow_html=True)

# ------------- LOAD DOCUMENT DATA --------------
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

# Sample metrics (replace with real DB logs if available)
SEARCHES_TODAY = 42
AVG_RESPONSE = "1.2s"
SUCCESS_RATE = "98.9%"
USERS_TODAY = 14
POPULAR_SEARCHES = ["Vacation Policy", "Work From Home", "Sick Leave", "Team Meetings", "Employee Handbook", "IT Support"]

# --------------- SIDEBAR ----------------
with st.sidebar:
    st.header("🛠 Admin Tools")
    if st.button("🔄 Refresh Documents"):
        with st.spinner("Updating documents from Google Drive..."):
            subprocess.run(["python", "update_embeddings.py"])
        st.success("✅ Documents refreshed!")

    st.markdown("### 📊 Quick Stats")
    st.markdown(f"""
    <div class='metric-card gradient-blue'>
        <div class='stat-number'>{DOCUMENT_COUNT}</div>
        <div class='stat-label'>Documents Indexed</div>
    </div>
    <div class='metric-card gradient-magenta'>
        <div class='stat-number'>{USERS_TODAY}</div>
        <div class='stat-label'>Active Users Today</div>
    </div>
    <div class='metric-card gradient-green'>
        <div class='stat-number'>{SUCCESS_RATE}</div>
        <div class='stat-label'>Success Rate</div>
    </div>
    """, unsafe_allow_html=True)

# ------------- METRIC CARDS (MAIN AREA) -------------
cols = st.columns(4)
with cols[0]:
    st.markdown(f"""
    <div class="metric-card gradient-blue">
        <div style="font-size:2em;">📄</div>
        <div class="stat-number">{DOCUMENT_COUNT}</div>
        Documents
    </div>
    """, unsafe_allow_html=True)
with cols[1]:
    st.markdown(f"""
    <div class="metric-card gradient-magenta">
        <div style="font-size:2em;">🔍</div>
        <div class="stat-number">{SEARCHES_TODAY}</div>
        Searches Today
    </div>
    """, unsafe_allow_html=True)
with cols[2]:
    st.markdown(f"""
    <div class="metric-card gradient-yellow">
        <div style="font-size:2em;">⚡</div>
        <div class="stat-number">{AVG_RESPONSE}</div>
        Avg Response
    </div>
    """, unsafe_allow_html=True)
with cols[3]:
    st.markdown(f"""
    <div class="metric-card gradient-green">
        <div style="font-size:2em;">✅</div>
        <div class="stat-number">{SUCCESS_RATE}</div>
        Success Rate
    </div>
    """, unsafe_allow_html=True)

# ------------- POPULAR SEARCH TAGS -------------
st.markdown("## <b>Quick Start - Popular Searches</b>", unsafe_allow_html=True)
tag_cols = st.columns(min(len(POPULAR_SEARCHES), 3))
for i, tag in enumerate(POPULAR_SEARCHES):
    tag_cols[i % len(tag_cols)].markdown(
        f"<button class='tag-btn'>{tag}</button>", unsafe_allow_html=True
    )

# ------------- TABS: SEARCH / FAQ / ABOUT -------------
tab_search, tab_faq, tab_about = st.tabs(["🔍 Search", "📖 FAQ", "ℹ About"])

# --------- MODEL LOAD ---------
model = None
index = None
if embedding_matrix is not None and len(embedding_matrix):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = create_faiss_index(embedding_matrix)

with tab_search:
    with st.form(key="search_form"):
        user_email = st.text_input(
            "Enter your Google email for document access:",
            key="email",
            placeholder="you@familytlc.com",
        )
        question = st.text_input(
            "What would you like to know today?",
            key="query",
            placeholder="Show me the sick leave policy, find training materials, etc.",
        )
        submitted = st.form_submit_button("Search Documents")

    if submitted:
        if not user_email:
            st.markdown('<div class="status-error">Please enter your email address for secure access.</div>', unsafe_allow_html=True)
        elif not question:
            st.markdown('<div class="status-warning">Please enter a question to search documents.</div>', unsafe_allow_html=True)
        elif model is None or index is None:
            st.markdown('<div class="status-error">Search not initialized. Please contact admin or refresh documents.</div>', unsafe_allow_html=True)
        else:
            SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']
            try:
                # Load Google credentials from environment variable
                service_account_info = json.loads(os.environ.get("GCP_SERVICE_ACCOUNT"))
                creds = service_account.Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
                drive_service = build('drive', 'v3', credentials=creds)

                def get_user_accessible_file_ids(email):
                    query = f"'{email}' in readers"
                    file_ids_set = set()
                    page_token = None
                    while True:
                        response = drive_service.files().list(
                            q=query, fields="nextPageToken, files(id)", pageToken=page_token
                        ).execute()
                        file_ids_set.update({f['id'] for f in response.get('files', [])})
                        page_token = response.get('nextPageToken', None)
                        if not page_token:
                            break
                    return file_ids_set

                allowed_file_ids = get_user_accessible_file_ids(user_email)
            except Exception as e:
                st.markdown(f'<div class="status-error">Google Drive error: {e}</div>', unsafe_allow_html=True)
                allowed_file_ids = set()

            # Save recent queries in session
            if 'recent_queries' not in st.session_state:
                st.session_state.recent_queries = []
            st.session_state.recent_queries.insert(0, {"question": question, "time": datetime.now().strftime("%Y-%m-%d %H:%M")})
            st.session_state.recent_queries = st.session_state.recent_queries[:10]

            st.markdown(f'<div class="status-success">Searching for: <b>{question}</b></div>', unsafe_allow_html=True)
            with st.spinner("🔎 Searching relevant documents..."):
                query_embedding = model.encode([question]).astype("float32")
                distances, indices = index.search(query_embedding, 10)
                results = []
                for i, idx in enumerate(indices[0]):
                    chunk = text_chunks[idx].lower()
                    score = distances[0][i]
                    results.append((score, idx))
                results.sort()
                top_indices = [idx for _, idx in results[:3]]

            if not top_indices:
                st.markdown('<div class="status-warning">No results. Try rewording or updating your filters.</div>', unsafe_allow_html=True)
            else:
                st.success(f"✅ Found {len(top_indices)} results.")
                for idx in top_indices:
                    file_name = sources[idx]
                    chunk = text_chunks[idx][:500].replace("\n", " ")
                    file_id = file_ids[idx]
                    view_link = f"https://drive.google.com/file/d/{file_id}/view"
                    if file_id in allowed_file_ids:
                        st.markdown(f"""
                        <div class="result-card">
                            <h4>📄 {file_name}</h4>
                            <p>{chunk}...</p>
                            <a href="{view_link}" target="_blank">🔗 View Document</a>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="status-error">🔒 No access to <b>{file_name}</b></div>', unsafe_allow_html=True)

with tab_faq:
    st.markdown("""
    <div class="result-card">
        <p><b>Q: What can I search here?</b><br>
        A: Any internal file, handbook, or HR document you are permitted to see on Google Drive.</p>
        <p><b>Q: Why do I enter my email address?</b><br>
        A: The bot verifies your Drive access to show only the documents you have permission to read.</p>
    </div>
    """, unsafe_allow_html=True)

with tab_about:
    st.markdown("""
    <div class="result-card">
        <b>FamilyTLC Search Bot</b> powers smarter, permission-aware search over your organization’s Google Drive.<br>
        <ul>
            <li>🔎 Semantic search: Sentence Transformers + FAISS</li>
            <li>🔒 Results filtered by your access rights</li>
            <li>📄 Auto-updates when documents change</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
