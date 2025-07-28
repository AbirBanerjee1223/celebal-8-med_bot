import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration ---
DATA_PATH = "data"          # Directory where your health documents are stored
DB_FAISS_PATH = "vectorstore/db_faiss" # Directory to save your FAISS vector database

# --- 1. Load Documents ---
def load_documents(data_path):
    """
    Loads all .txt and .pdf documents from the specified data path.
    """
    documents = []
    for root, _, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".txt"):
                print(f"Loading Text file: {file_path}")
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
            elif file.endswith(".pdf"):
                print(f"Loading PDF file: {file_path}")
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
    return documents

# --- 2. Split Documents into Chunks ---
def split_documents(documents):
    """
    Splits loaded documents into smaller, overlapping chunks.
    """
    # RecursiveCharacterTextSplitter attempts to split by paragraphs, then sentences, then words
    # This helps keep related text together.
    # chunk_size: maximum size of each chunk (in characters)
    # chunk_overlap: number of characters to overlap between chunks to maintain context
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(texts)} chunks.")
    return texts

# --- 3. Create Embeddings and Build FAISS Vector Store ---
def create_embeddings_and_vector_store(texts, db_faiss_path):
    """
    Generates embeddings for text chunks and creates/saves a FAISS vector store.
    """
    # Use a pre-trained Sentence Transformer model for embeddings
    # all-MiniLM-L6-v2 is a good balance of performance and speed/size for local use
    print("Initializing embedding model: sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Creating FAISS vector store...")
    # Create the FAISS vector store from the document chunks and their embeddings
    db = FAISS.from_documents(texts, embeddings)
    print("FAISS vector store created.")

    # Save the vector store to disk for later use
    os.makedirs(db_faiss_path, exist_ok=True) # Ensure the directory exists
    db.save_local(db_faiss_path)
    print(f"FAISS vector store saved locally at: {db_faiss_path}")

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting document processing and vector DB creation...")
    
    # 1. Load documents
    loaded_documents = load_documents(DATA_PATH)
    if not loaded_documents:
        print(f"No documents found in {DATA_PATH}. Please ensure your .txt and .pdf files are there.")
        print("Exiting.")
    else:
        # 2. Split documents
        text_chunks = split_documents(loaded_documents)

        # 3. Create embeddings and build/save FAISS DB
        create_embeddings_and_vector_store(text_chunks, DB_FAISS_PATH)
        print("Process complete. FAISS vector database is ready!")