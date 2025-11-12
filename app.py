import streamlit as st
import os
import json
from scripts import config
from scripts.extract_text import extract_pdf_text
from scripts.clean_text import process_file
from scripts.embed_chunks import create_embeddings
from scripts.retrieve import load_vector_store, retrieve_chunks
from scripts.qa_chatbot import ask_question

# Ensure data directories exist
os.makedirs(config.PROCESSED_DIR, exist_ok=True)
os.makedirs(config.CHUNKS_DIR, exist_ok=True)
os.makedirs(config.EMBEDDINGS_DIR, exist_ok=True)

st.set_page_config(page_title="Document QA System", page_icon="📘", layout="wide")
st.title("📘 Document Q&A System")
st.markdown("Upload a PDF, process it into searchable chunks, and ask questions about its content.")

# --- Step 1: PDF Upload ---
uploaded_pdf = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_pdf:
    pdf_path = os.path.join(config.DATA_DIR, "raw", uploaded_pdf.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.read())

    base_name = os.path.splitext(uploaded_pdf.name)[0]
    processed_text_path = os.path.join(config.PROCESSED_DIR, f"{base_name}.txt")
    chunks_path = os.path.join(config.CHUNKS_DIR, f"{base_name}_chunks.json")
    embeddings_dir = config.EMBEDDINGS_DIR

    # --- Step 2: Run pipeline stages ---
    if st.button(" Run Processing Pipeline"):
        with st.spinner("Extracting text from PDF..."):
            text = extract_pdf_text(pdf_path, processed_text_path)

        # ✅ Friendly error if no text found
        if not text or not text.strip():
            st.error("⚠️ Text could not be extracted from this PDF. It may be a scanned document or contain only images.")
            st.stop()

        with st.spinner("Cleaning and chunking text..."):
            chunks = process_file(processed_text_path, config.CHUNKS_DIR, chunk_length=config.CHUNK_SIZE)

        with st.spinner("Generating embeddings (OpenAI or fallback)..."):
            create_embeddings(chunks, index_dir=config.EMBEDDINGS_DIR, index_name=config.INDEX_NAME)

        st.success("✅ Processing complete! Your document is ready for Q&A.")

        st.write(f"**Processed text saved at:** `{processed_text_path}`")
        st.write(f"**Chunks saved at:** `{chunks_path}`")
        st.write(f"**Embeddings stored in:** `{config.EMBEDDINGS_DIR}`")

# --- Step 3: Q&A Interface ---
st.markdown("---")
st.subheader(" Ask Questions About Your Document")

if os.path.exists(os.path.join(config.EMBEDDINGS_DIR, config.INDEX_NAME)):
    vector_store = load_vector_store(os.path.join(config.EMBEDDINGS_DIR, config.INDEX_NAME))

    user_query = st.text_input("Enter your question:")
    if st.button("Ask"):
        with st.spinner("Retrieving and generating answer..."):
            answer = ask_question(vector_store, user_query)
        st.success("Answer:")
        st.write(answer)
else:
    st.info("Please process a document first before asking questions.")
