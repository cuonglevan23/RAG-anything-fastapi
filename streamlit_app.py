import streamlit as st
import requests
import time
import os
from pathlib import Path

# Configuration
# Default to local, but you should paste your Ngrok URL here (e.g., https://xxx.ngrok-free.app)
API_BASE_URL = "https://a5af-34-143-214-246.ngrok-free.app"
API_PREFIX = "/api/v1/rag"

st.set_page_config(
    page_title="RAG Anything UI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #262730;
        color: white;
        border: 1px solid #4a4a4a;
    }
    .stButton>button:hover {
        border-color: #ff4b4b;
        color: #ff4b4b;
    }
    .status-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #1e1e1e;
        border: 1px solid #333;
        margin-bottom: 20px;
    }
    .log-container {
        font-family: 'Courier New', Courier, monospace;
        background-color: #000;
        color: #0f0;
        padding: 10px;
        border-radius: 5px;
        height: 200px;
        overflow-y: auto;
        font-size: 0.8em;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 RAG Anything - Production UI")
st.markdown("---")

# Sidebar for configuration and info
with st.sidebar:
    st.header("⚙️ Settings")
    base_url = st.text_input("Ngrok/API Host", value=API_BASE_URL)
    api_url = f"{base_url.rstrip('/')}{API_PREFIX}"
    st.info(f"API Endpoint: {api_url}")
    
    st.markdown("---")
    st.header("🗂️ Workspace Manager")
    
    # Fetch projects
    try:
        proj_resp = requests.get(f"{api_url}/projects")
        if proj_resp.status_code == 200:
            existing_projects = proj_resp.json().get("projects", [])
        else:
            existing_projects = []
    except:
        existing_projects = []

    # New Project
    new_project = st.text_input("Create New Project", placeholder="e.g. biology_book")
    if st.button("➕ Create"):
        if new_project:
            # Simply selecting it will create it on first upload/query in backend
            st.session_state.project_id = new_project
            st.success(f"Project '{new_project}' ready!")
            st.rerun()

    # Ensure currently selected project is in list even if not on disk yet
    if "project_id" in st.session_state and st.session_state.project_id not in existing_projects:
        existing_projects.append(st.session_state.project_id)

    # Select Project
    if existing_projects:
        # Sort for better UI
        existing_projects.sort()
        
        selected_project = st.selectbox(
            "Select Workspace", 
            options=existing_projects,
            index=existing_projects.index(st.session_state.project_id) if st.session_state.get("project_id") in existing_projects else 0
        )
        st.session_state.project_id = selected_project
    else:
        st.warning("No projects found. Create one above!")
        if "project_id" not in st.session_state:
            st.session_state.project_id = "default"

    st.markdown(f"**Current Workspace:** `{st.session_state.project_id}`")
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Tabs for different functionalities
tab1, tab2 = st.tabs(["📤 Upload & Process", "💬 Chat with RAG"])

with tab1:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Choose a file (PDF, TXT, DOCX, MD, Images)", type=["pdf", "txt", "docx", "md", "png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        if st.button("🚀 Start Processing"):
            with st.status("Uploading file...", expanded=True) as status:
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data_payload = {"project_id": st.session_state.project_id}
                    response = requests.post(f"{api_url}/upload", files=files, data=data_payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        task_id = data["task_id"]
                        st.success(f"File uploaded! Task ID: {task_id}")
                        
                        # Polling for status
                        progress_bar = st.progress(0, text="Initializing processing...")
                        log_placeholder = st.empty()
                        
                        while True:
                            status_resp = requests.get(f"{api_url}/status/{task_id}")
                            if status_resp.status_code == 200:
                                s_data = status_resp.json()
                                percentage = s_data.get("percentage", 0) / 100
                                current_status = s_data.get("status")
                                logs = s_data.get("logs", [])
                                
                                progress_bar.progress(percentage, text=f"Status: {s_data.get('message')}")
                                
                                # Show logs
                                with log_placeholder.container():
                                    st.markdown("**Processing Logs:**")
                                    log_html = "".join([f"<div>> {log}</div>" for log in logs])
                                    st.markdown(f'<div class="log-container">{log_html}</div>', unsafe_allow_html=True)
                                
                                if current_status == "completed":
                                    st.balloons()
                                    status.update(label="Processing Complete!", state="complete", expanded=False)
                                    break
                                elif current_status == "failed":
                                    st.error(f"Processing failed: {s_data.get('error')}")
                                    status.update(label="Processing Failed", state="error", expanded=True)
                                    break
                            
                            time.sleep(2) # Poll every 2 seconds
                    else:
                        st.error(f"Error: {response.text}")
                        status.update(label="Upload Failed", state="error")
                except Exception as e:
                    st.error(f"Connection Error: {str(e)}")
                    status.update(label="Connection Error", state="error")

with tab2:
    st.header("Query the RAG Engine")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask anything about your documents..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    payload = {
                        "query": prompt, 
                        "mode": "hybrid",
                        "project_id": st.session_state.project_id
                    }
                    response = requests.post(f"{api_url}/query", json=payload)
                    
                    if response.status_code == 200:
                        full_response = response.json().get("response", "No response received.")
                        st.markdown(full_response)
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                    else:
                        st.error(f"API Error: {response.status_code}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("Powered by RAG Anything, Mineru & LightRAG")
