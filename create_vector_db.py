import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data"
DB_FAISS_PATH = "vectorstore/db_faiss"

def load_documents(data_path):
    documents = []
    for root, _, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".txt"):
                print(f"Loading: {file_path}")
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
            elif file.endswith(".pdf"):
                print(f"Loading: {file_path}")
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(texts)} chunks.")
    return texts

def create_embeddings_and_vector_store(texts, db_path):
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Building FAISS vector store...")
    db = FAISS.from_documents(texts, embeddings)

    os.makedirs(db_path, exist_ok=True)
    db.save_local(db_path)
    print(f"Vector store saved at: {db_path}")

if __name__ == "__main__":
    print("Processing documents...")

    docs = load_documents(DATA_PATH)
    if not docs:
        print(f"No documents found in '{DATA_PATH}'.")
    else:
        chunks = split_documents(docs)
        create_embeddings_and_vector_store(chunks, DB_FAISS_PATH)
        print("Vector store ready.")
