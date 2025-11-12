#extract_text.py
#!/usr/bin/env python3
import sys
import pdfplumber
import os

def extract_pdf_text(pdf_path, output_file):
    """
    Extract text from a PDF and save it to a text file.

    Args:
        pdf_path (str): Absolute path to the PDF file.
        output_file (str): Absolute path to save the extracted text.
    """
    # Ensure the PDF exists
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # some pages may be empty
                text += page_text + "\n"

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write extracted text to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

    return text

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 extract_text.py <PDF_PATH> <OUTPUT_FILE>")
        sys.exit(1)

    pdf_file = sys.argv[1]
    output_file = sys.argv[2]

    extracted_text = extract_pdf_text(pdf_file, output_file)
    print(f" Extraction complete. Text saved to {output_file}")
