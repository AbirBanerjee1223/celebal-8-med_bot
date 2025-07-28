import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings # Keep this for FAISS embeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import google.generativeai as genai

# Import GoogleGenerativeAI for the LLM
from langchain_google_genai import GoogleGenerativeAI # <--- ADD THIS IMPORT

import traceback # For detailed error messages

# --- Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # <--- Get the Google API key


DB_FAISS_PATH = "vectorstore/db_faiss"
# No specific MODEL_NAME needed here for Gemini API as it's defined in the LLM class directly

# --- 1. Load the FAISS Vector Store ---
# (This part is crucial and remains the same. Make sure HuggingFaceEmbeddings import is correct)
def load_faiss_vector_store(db_path):
    """
    Loads the FAISS vector store from the specified path.
    The embeddings model used for loading must be the same as the one used for creation.
    """
    print("Loading embedding model for FAISS...")
    # Fix the deprecation warning for HuggingFaceEmbeddings here if you haven't already:
    # from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print(f"Loading FAISS vector store from: {db_path}")
    db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    print("FAISS vector store loaded successfully.")
    return db

# --- 2. Load the LLM (Now using Google Gemini API) ---
def load_gemini_llm(api_key):
    """
    Initializes and returns the Google Gemini LLM via API.
    """
    if not api_key:
        raise ValueError("Google API Key not found. Please set GOOGLE_API_KEY in your .env file.")

    print("Initializing Google Gemini LLM via API...")
    llm = GoogleGenerativeAI(
        model="models/gemini-1.5-flash", # Use 'gemini-pro' for text-based tasks
        google_api_key=api_key,
        temperature=0.1,    # Keep low for factual Q&A
        max_output_tokens=512 # Adjust based on your expected answer length
    )
    print("Google Gemini LLM initialized.")
    return llm

# --- 3. Define the RAG Prompt Template ---
# (This part remains exactly the same)
def create_rag_prompt_template():
    """
    Defines the prompt template that combines retrieved context with the user's question.
    """
    template = """Use the following pieces of context to answer the user's question.
If you don't know the answer based on the provided context, politely state that you don't have enough information from the given knowledge base.
Do not make up an answer.
This information is for general knowledge and should not replace professional medical advice.

Context:
{context}

Question:
{question}

Answer:"""
    return PromptTemplate(template=template, input_variables=["context", "question"])

# --- Main Chatbot Application ---
def run_chatbot():
    """
    Main function to run the RAG Q&A chatbot.
    """
    print("\n--- Initializing Healthcare Info Bot ---")
    
    # Load FAISS
    vector_db = load_faiss_vector_store(DB_FAISS_PATH)
    
    # Create a retriever from the FAISS database
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # Load Gemini LLM
    llm = load_gemini_llm(GOOGLE_API_KEY) # <--- Use the Gemini loading function

    # Create the RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": create_rag_prompt_template()}
    )
    
    print("\n--- Healthcare Info Bot Ready! Type 'exit' to quit. ---")
    
    while True:
        user_query = input("\nYour Question: ")
        if user_query.lower() == 'exit':
            print("Exiting chatbot. Goodbye!")
            break
        
        if not user_query.strip():
            print("Please enter a question.")
            continue

        print("Thinking...")
        try:
            result = qa_chain.invoke({"query": user_query})
            
            print("\nBot Answer:")
            print(result["result"])

            print("\n--- Sources Used ---")
            if result.get("source_documents"):
                for i, doc in enumerate(result["source_documents"]):
                    print(f"Source {i+1}:")
                    print(f"  Content (excerpt): {doc.page_content[:200]}...")
                    if doc.metadata:
                        print(f"  Metadata: {doc.metadata}")
            else:
                print("No specific source documents retrieved for this query.")
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc() # Print full traceback
            print("\nPlease try rephrasing your question or check the API key and internet connection.")

if __name__ == "__main__":
    run_chatbot()