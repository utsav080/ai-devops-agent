from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os

def build_vector_store_from_docs(folder_path, hf_token=None):  # token no longer used
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".txt", ".md", ".py", ".yml", ".yaml")) or filename == "Dockerfile":
            path = os.path.join(folder_path, filename)
            loader = TextLoader(path)
            documents.extend(loader.load())

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)

    # ✅ Removed invalid token parameter
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    return FAISS.from_documents(split_docs, embeddings)

def get_context_from_docs(vs, query):
    docs = vs.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])
