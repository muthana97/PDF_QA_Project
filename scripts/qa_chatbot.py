#scripts/qa_chatbot.py
import os
import json
from scripts.config import API_KEY, BASE_URL, USE_GROQ, MODEL_QA
from scripts.retrieve import load_vector_store, retrieve_chunks

# Handle both Groq and OpenAI dynamically
try:
    if USE_GROQ:
        from groq import Groq
        client = Groq(api_key=API_KEY)
    else:
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
except Exception as e:
    print(f" Error initializing client: {e}")
    exit(1)


def ask_question(vector_store, question, top_k=3):
    """
    Retrieves the most relevant chunks and generates an answer using the selected LLM.
    """
    # Retrieve top chunks
    relevant_chunks = retrieve_chunks(vector_store, question, top_k=top_k)
    if not relevant_chunks:
        return "I couldn’t find relevant information in the database."

    # Build context
    context_text = "\n\n".join(relevant_chunks)
    prompt = f"""
    You are an AI assistant that answers questions based strictly on the provided context.
    If the answer is not in the context, respond with: "I don't have enough information."
    
    Context:
    {context_text}
    
    Question: {question}
    Answer:
    """

    # Generate response
    if USE_GROQ:
        response = client.chat.completions.create(
            model=MODEL_QA,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    else:
        response = client.chat.completions.create(
            model=MODEL_QA,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()


# Example usage
if __name__ == "__main__":
    print(f" Using {'Groq' if USE_GROQ else 'OpenAI'} backend ({MODEL_QA})")

    vector_store_path = "data/embeddings/faiss_index"
    if not os.path.exists(vector_store_path):
        print(f" FAISS index not found at {vector_store_path}. Please run embed_chunks.py first.")
        exit(1)

    vs = load_vector_store(vector_store_path)

    while True:
        question = input("\n Enter your question (or type 'exit'): ").strip()
        if question.lower() in ["exit", "quit"]:
            print(" Exiting chatbot.")
            break

        print(" Retrieving and generating answer...")
        answer = ask_question(vs, question)
        print("\n Answer:", answer)
