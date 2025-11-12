#clean_text.py
#!/usr/bin/env python3
import re
import os
import json
import sys

def clean_text(text):
    """
    Cleans the raw text:
    - Removes extra spaces and line breaks
    - Removes non-ASCII characters
    - Normalizes spacing
    """
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Replace multiple spaces/newlines with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, max_length=800):
    """
    Splits text into chunks of max_length characters
    """
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def process_file(input_path, output_dir, chunk_length=800):
    # Ensure input exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Read raw text
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    # Clean
    cleaned = clean_text(raw_text)
    
    # Chunk
    chunks = chunk_text(cleaned, max_length=chunk_length)
    
    # Ensure output folder exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save chunks as JSON
    base_name = os.path.basename(input_path).replace('.txt', '')
    output_path = os.path.join(output_dir, f"{base_name}_chunks.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2)
    
    print(f" Processed {input_path}, saved {len(chunks)} chunks to {output_path}")
    return chunks

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 clean_text.py <INPUT_FILE> <OUTPUT_FOLDER>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_folder = sys.argv[2]

    process_file(input_file, output_folder)
