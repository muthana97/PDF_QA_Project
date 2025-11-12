# scripts/retrieve.py
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_vector_store(index_path="data/embeddings/faiss_index"):
    """
    Loads FAISS index. 
    Attempts to use OpenAIEmbeddings first (if OPENAI_API_KEY present), 
    then falls back to HuggingFaceEmbeddings.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index directory not found: {index_path}")

    # Try OpenAI embeddings first
    embedding_model = None
    try:
        #from langchain_community.embeddings import OpenAIEmbeddings
        #import openai, os
        if os.getenv("OPENAI_API_KEY"):
            print(" Trying to load FAISS index with OpenAI embeddings...")
            embedding_model = OpenAIEmbeddings()
    except Exception:
        embedding_model = None

    # Fallback to HuggingFace
    if embedding_model is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print(" Falling back to HuggingFace embeddings (sentence-transformers/all-MiniLM-L6-v2)...")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    print(" Successfully loaded FAISS index.")
    return vector_store


def retrieve_chunks(vector_store, query, top_k=3):
    """
    Finds the top_k most relevant chunks to the user query.
    Returns a list of text strings (page_content).
    """
    results = vector_store.similarity_search(query, k=top_k)
    return [r.page_content for r in results]


# Example usage
if __name__ == "__main__":
    q = "What are the main leadership theories discussed?"
    vs = load_vector_store()
    chunks = retrieve_chunks(vs, q)
    for i, c in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ---\n{c}\n")
