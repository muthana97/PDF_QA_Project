#!/usr/bin/env python3
import os
import sys
import subprocess

# Ensure we can import from 'scripts' no matter where this runs from
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
sys.path.insert(0, PROJECT_ROOT)

from scripts import config  # consistent import

# Define relative paths to all pipeline scripts
SCRIPTS = {
    "extract": os.path.join(SCRIPTS_DIR, "extract_text.py"),
    "clean": os.path.join(SCRIPTS_DIR, "clean_text.py"),
    "embed": os.path.join(SCRIPTS_DIR, "embed_chunks.py"),
    "retrieve": os.path.join(SCRIPTS_DIR, "retrieve.py"),
    "qa": os.path.join(SCRIPTS_DIR, "qa_chatbot.py"),
}


def run_stage(stage_name, *args):
    """Run one stage script with correct environment keys"""
    script_path = SCRIPTS.get(stage_name)
    if not script_path or not os.path.exists(script_path):
        print(f" Error: {stage_name} script not found at {script_path}")
        sys.exit(1)

    # Import provider configuration
    from scripts.config import USE_GROQ, OPENAI_API_KEY, GROQ_API_KEY, API_KEY, BASE_URL

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = OPENAI_API_KEY or ""
    env["GROQ_API_KEY"] = GROQ_API_KEY or ""
    env["USE_GROQ"] = str(USE_GROQ)
    env["API_KEY"] = API_KEY
    env["BASE_URL"] = BASE_URL

    print(f"\n Running {stage_name} stage ({'Groq' if USE_GROQ else 'OpenAI'})...")
    subprocess.run(["python3", script_path] + list(args), check=True, env=env)


def main(pdf_path):
    """Run the entire FAISS pipeline end-to-end"""
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    processed_text_path = os.path.join(config.PROCESSED_DIR, f"{base_name}.txt")
    chunks_path = os.path.join(config.CHUNKS_DIR, f"{base_name}_chunks.json")
    embeddings_dir = config.EMBEDDINGS_DIR

    # Stage 1: Extract text
    run_stage("extract", pdf_path, processed_text_path)

    # Stage 2: Clean & chunk
    run_stage("clean", processed_text_path, config.CHUNKS_DIR)

    # Stage 3: Create embeddings
    run_stage("embed", chunks_path, embeddings_dir)

    print("\n Data prepared. You can now run QA manually:")
    print(f"python scripts/qa_chatbot.py")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py path_to_pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    main(pdf_path)
