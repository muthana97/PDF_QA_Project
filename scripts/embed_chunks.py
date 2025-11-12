#scripts/embed_chunks.py
import json
import sys
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from scripts.config import MODEL_EMBEDDING
from openai import RateLimitError, AuthenticationError, APIError

def create_embeddings(chunks, index_dir="data/embeddings", index_name="faiss_index"):
    docs = [Document(page_content=c) for c in chunks]

    # Always use OpenAI for embeddings first
    embedding_key = os.getenv("OPENAI_API_KEY")
    embedding_base_url = "https://api.openai.com/v1"

    if not embedding_key:
        print("No OpenAI API key found in environment. Falling back to Hugging Face embeddings.")
        hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(docs, hf_embeddings)
        vector_store.save_local(os.path.join(index_dir, index_name))
        print(f"Embeddings created and saved to {os.path.join(index_dir, index_name)}")
        return vector_store

    # Initialize embedding model
    try:
        print(f"Using OpenAI embeddings ({MODEL_EMBEDDING}) at {embedding_base_url}")
        embeddings = OpenAIEmbeddings(
            model=MODEL_EMBEDDING,
            api_key=embedding_key,
            base_url=embedding_base_url,
        )

        os.makedirs(index_dir, exist_ok=True)
        print("Building FAISS index using OpenAI embeddings...")
        vector_store = FAISS.from_documents(docs, embeddings)

    except (RateLimitError, AuthenticationError, APIError, Exception) as e:
        print(f"Embedding failed with OpenAI: {e}")
        print("Falling back to Hugging Face embeddings (sentence-transformers/all-MiniLM-L6-v2)")
        hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(docs, hf_embeddings)

    # Save FAISS index
    vector_store.save_local(os.path.join(index_dir, index_name))
    print(f"Embeddings created and saved to {os.path.join(index_dir, index_name)}")

    return vector_store


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 scripts/embed_chunks.py <chunks_path> <output_dir>")
        sys.exit(1)

    chunks_path = sys.argv[1]
    index_dir = sys.argv[2]

    print("Embedding stage: using OpenAI as primary provider (Groq does not support embeddings).")

    # Load chunks
    if not os.path.exists(chunks_path):
        print(f"Chunks file not found: {chunks_path}")
        sys.exit(1)

    with open(chunks_path, "r") as f:
        chunks_data = json.load(f)

    if isinstance(chunks_data[0], dict) and "content" in chunks_data[0]:
        chunks = [c["content"] for c in chunks_data]
    else:
        chunks = chunks_data

    # Create embeddings
    create_embeddings(chunks, index_dir=index_dir)
