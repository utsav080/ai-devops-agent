import streamlit as st
import os
import requests
from langchain_community.embeddings import HuggingFaceEmbeddings
from ask_llm_direct import ask_llm
from rag_utils import build_vector_store_from_docs, get_context_from_docs

# Load secrets
HUGGINGFACEHUB_API_TOKEN = st.secrets["HF_API_KEY"]
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
GITHUB_REPO = st.secrets["GITHUB_REPO"]
GITHUB_WORKFLOW = st.secrets["GITHUB_WORKFLOW"]

# Initialize session state
for key in ["summary", "dockerfile", "workflow"]:
    if key not in st.session_state:
        st.session_state[key] = None

# UI
st.title("🤖 AI DevOps Agent (Free Tier - Hugging Face)")

# Upload docs
uploaded_files = st.file_uploader("📁 Upload Docs", type=["txt", "md", "py", "yml", "yaml", "Dockerfile"], accept_multiple_files=True)
if uploaded_files:
    os.makedirs("docs", exist_ok=True)
    for file in uploaded_files:
        with open(os.path.join("docs", file.name), "wb") as f:
            f.write(file.getbuffer())
    st.success(f"✅ Uploaded {len(uploaded_files)} file(s)")

# Paste release notes
release_notes = st.text_area("📄 Paste Release Notes", height=200)

if st.button("🔍 Understand & Generate DevOps Files") and release_notes:
    st.info("⏳ Reading documents...")

    vs = build_vector_store_from_docs("docs", HUGGINGFACEHUB_API_TOKEN)
    context = get_context_from_docs(vs, release_notes)

    # Summary
    summary_prompt = f"""
You are an expert DevOps engineer.

The following is a software release description and a piece of code.

Your job is to:
1. Summarize what the code does in plain English.
2. Identify key deployment-relevant features (e.g., use of Flask, JWT, SQLAlchemy).
3. Suggest how such a backend could be deployed.

### Release Notes:
{release_notes}

### Code:
{context}

### Output (Summary + Deployment Steps):
"""
    summary = ask_llm(summary_prompt, HUGGINGFACEHUB_API_TOKEN)
    st.session_state.summary = summary

    # Dockerfile
    docker_prompt = f"""Based on this project, generate a Dockerfile:

Summary:
{summary}"""
    dockerfile = ask_llm(docker_prompt, HUGGINGFACEHUB_API_TOKEN)
    st.session_state.dockerfile = dockerfile
    os.makedirs("output", exist_ok=True)
    with open("output/Dockerfile", "w") as f:
        f.write(dockerfile)

    # GitHub Actions
    gh_prompt = f"""Generate a GitHub Actions workflow (deploy.yml) that:
- Builds Docker image from Dockerfile
- Pushes to Docker Hub
- Deploys to staging

Summary:
{summary}"""
    workflow = ask_llm(gh_prompt, HUGGINGFACEHUB_API_TOKEN)
    st.session_state.workflow = workflow
    with open("output/deploy.yml", "w") as f:
        f.write(workflow)

# Output section (only visible after generation)
if st.session_state.summary:
    st.subheader("🧠 Deployment Summary")
    st.write(st.session_state.summary)

    st.download_button("📦 Download Dockerfile", st.session_state.dockerfile, file_name="Dockerfile")
    st.download_button("⚙️ Download GitHub Workflow", st.session_state.workflow, file_name="deploy.yml")

# GitHub Deployment Trigger
st.divider()
st.subheader("🚀 Deploy to Staging")
if st.button("▶️ Trigger Deployment"):
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    trigger_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{GITHUB_WORKFLOW}/dispatches"
    payload = {"ref": "main"}

    res = requests.post(trigger_url, headers=headers, json=payload)
    if res.status_code == 204:
        st.success("✅ Workflow triggered!")
    else:
        st.error(f"❌ Failed to trigger workflow. Status: {res.status_code}")
        st.code(res.text)

# Workflow Status
st.divider()
if st.button("🔄 Refresh Deployment Status"):
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs"
    r = requests.get(url, headers=headers)

    if r.status_code == 200:
        runs = r.json().get("workflow_runs", [])
        for run in runs[:3]:
            st.write(f"🔁 **{run['name']}** — {run['status']} / {run['conclusion']}")
            st.markdown(f"[🔗 View on GitHub]({run['html_url']})")
    else:
        st.error("❌ Could not fetch status.")
