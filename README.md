# PDF RAG Chatbot Pipeline

This project implements an end-to-end Retrieval-Augmented Generation (RAG) pipeline for working with PDF documents. It extracts, cleans, embeds, and retrieves contextual answers using FAISS, LangChain, and either the Groq or OpenAI API for language model inference.

The goal is to provide a simple and modular framework that demonstrates RAG principles, while allowing flexibility in backend providers and local embeddings.


# Features

- Extracts and processes text from PDF files.
- Cleans and chunks text for efficient retrieval.
- Builds embeddings using OpenAI (with fallback to Hugging Face models).
- Uses FAISS for vector storage and similarity search.
- Runs question-answering using either Groq or OpenAI models.
- Modular structure allowing independent execution of each pipeline stage.


# Project Structure
├── LICENSE
├── README.md
├── app.py
├── changes.txt
├── data
│   ├── chunks
│   ├── embeddings
│   │   └── faiss_index
│   ├── processed
│   └── raw
├── processed
├── requirements.txt
├── run_pipeline.py
└── scripts
    ├── __init__.py
    ├── clean_text.py
    ├── config.py
    ├── embed_chunks.py
    ├── extract_text.py
    ├── qa_chatbot.py
    └── retrieve.py

SETUP
1) Clone the Repository
```bash
git clone https://github.com/yourusername/pdf-rag-chatbot.git
cd pdf-rag-chatbot

2) Create and activate a virtual environment by running the following bash commands
 depending on Operating system     

python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

3. Install dependencies
pip install -r requirements.txt

4. Configure environment variables 
# Groq mode
USE_GROQ=True #Set this to False if you would like to use an OPENAI API
GROQ_API_KEY=your_groq_key_here
BASE_URL="https://api.groq.com/openai/v1"
MODEL_QA="llama-3.3-70b-versatile"

# OpenAI (for embeddings)
OPENAI_API_KEY=your_openai_key_here
MODEL_EMBEDDING="text-embedding-3-small"

# Usage
1) Run the full pipeline
python run_pipeline.py path/to/your.pdf

2) Start the chatbot
python scripts/qa_chatbot.py

#Example interaction:

Enter your question (or type 'exit'): What is the main topic of the document?
Answer: The document discusses business administration principles.

# Technologies used
1- Python 3.14
2- LangChain
3- FAISS
4- Hugging Face Transformers
5- pdfplumber
6- Groq API / OpenAI API
7- dotenv for configuration management

# Next Steps
Planned improvements include:

1- Adding a simple web interface (Streamlit or React frontend)
2- Supporting multiple document uploads and retrieval
3- Containerizing the project with Docker
4- Improved logging and error handling


# Author

Almuthana Babiker
AI Engineer / Python Developer
GitHub: https://github.com/muthana97
LinkedIn: https://www.linkedin.com/in/almuthana-babiker-572098153/ 
 
# License

This project is open-source and available under the MIT License.

© 2025 Muthana. Released under the MIT License.

