import os
import traceback
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

DB_FAISS_PATH = "vectorstore/db_faiss"

def load_faiss_vector_store(db_path):
    print("Loading FAISS vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    print("FAISS vector store loaded.")
    return db

def load_gemini_llm(api_key):
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY in environment.")
    
    print("Initializing Gemini model...")
    llm = GoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.1,
        max_output_tokens=512
    )
    return llm

def create_rag_prompt_template():
    template = """Use the following pieces of context to answer the user's question.
If the answer is not in the context, say you don't have enough information from the provided knowledge base.

Context:
{context}

Question:
{question}

Answer:"""
    return PromptTemplate(template=template, input_variables=["context", "question"])

def run_chatbot():
    print("\n--- Healthcare Info Bot ---")
    
    vector_db = load_faiss_vector_store(DB_FAISS_PATH)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    llm = load_gemini_llm(GOOGLE_API_KEY)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": create_rag_prompt_template()}
    )
    
    print("\nBot is ready. Type 'exit' to quit.")
    
    while True:
        user_query = input("\nYour Question: ")
        if user_query.lower() == 'exit':
            print("Goodbye!")
            break

        if not user_query.strip():
            print("Please enter a valid question.")
            continue

        try:
            print("Thinking...")
            result = qa_chain.invoke({"query": user_query})

            print("\nBot Answer:")
            print(result["result"])

            print("\n--- Sources ---")
            if result.get("source_documents"):
                for i, doc in enumerate(result["source_documents"]):
                    print(f"Source {i+1}:")
                    print(f"  Content (excerpt): {doc.page_content[:200]}...")
                    if doc.metadata:
                        print(f"  Metadata: {doc.metadata}")
            else:
                print("No source documents retrieved.")
        
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    run_chatbot()
