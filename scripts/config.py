#scripts/config.py
import os
from dotenv import load_dotenv

# Project Root & .env Setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(dotenv_path)

# Load API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Determine Provider for QA Stage
USE_GROQ = os.getenv("USE_GROQ", "False").lower() == "true"

if USE_GROQ:
    print("Using Groq mode")
    API_KEY = GROQ_API_KEY
    BASE_URL = "https://api.groq.com/openai/v1"
    MODEL_QA = os.getenv("MODEL_QA", "llama-3.1-70b-versatile")
else:
    print("Using OpenAI mode")
    API_KEY = OPENAI_API_KEY
    BASE_URL = "https://api.openai.com/v1"
    MODEL_QA = os.getenv("MODEL_QA", "gpt-4o-mini")

# Validate keys
if USE_GROQ and not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing in .env but USE_GROQ=True")
if not USE_GROQ and not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY missing in .env and USE_GROQ=False")

# Directory Paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")

# Default Parameters
CHUNK_SIZE = 800
INDEX_NAME = "faiss_index"

# Embedding Model
# Always use OpenAI for embeddings (Groq does not provide embedding models)
MODEL_EMBEDDING = os.getenv("MODEL_EMBEDDING", "text-embedding-3-small")

# --- Helper function to print provider info ---
def print_provider_summary():
    print("\nCurrent configuration summary:")
    print(f" - QA Provider: {'Groq' if USE_GROQ else 'OpenAI'}")
    print(f" - BASE_URL: {BASE_URL}")
    print(f" - QA Model: {MODEL_QA}")
    print(f" - Embedding Model (always OpenAI): {MODEL_EMBEDDING}")
    print(f" - Key prefix: {API_KEY[:3]}*** (hidden for safety)\n")
