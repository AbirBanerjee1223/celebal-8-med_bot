import os
import traceback
from dotenv import load_dotenv
import streamlit as st

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAI

# --- Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_FAISS_PATH = "vectorstore/db_faiss"

# --- Loaders ---
@st.cache_resource
def load_faiss_vector_store(db_path):
    with st.spinner("Loading knowledge base..."):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    st.success("Knowledge base loaded!")
    return db

@st.cache_resource
def load_gemini_llm(api_key):
    if not api_key:
        st.error("Missing Google API Key. Check .env.")
        st.stop()

    with st.spinner("Initializing AI model..."):
        llm = GoogleGenerativeAI(
            model="models/gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.1,
            max_output_tokens=512
        )
    st.success("AI model ready.")
    return llm

# --- Setup Prompt & Chain ---
@st.cache_resource
def setup_qa_chain():
    vector_db = load_faiss_vector_store(DB_FAISS_PATH)
    llm = load_gemini_llm(GOOGLE_API_KEY)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    prompt = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template="""Use the following context to answer the question. 
If the answer is not in the context, say so without guessing.

Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:"""
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return chain

# --- Streamlit App UI ---
st.set_page_config(page_title="Local Healthcare Info Bot", page_icon="⚕️")
st.title("⚕️ Local Healthcare Info Bot")
st.markdown("Ask questions on common healthcare topics from the knowledge base.")

qa_chain = setup_qa_chain()

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    qa_chain.memory.clear()
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response_chunks = []

        try:
            for chunk in qa_chain.stream({"question": prompt}):
                if "answer" in chunk:
                    response_chunks.append(chunk["answer"])
                    message_placeholder.markdown("".join(response_chunks) + "▌")

            final_answer = "".join(response_chunks)
            message_placeholder.markdown(final_answer)

            # Fetch sources (approximation since not from streamed output directly)
            docs = qa_chain.retriever.get_relevant_documents(prompt)
            sources = "\n\n**Sources Used:**\n"
            for i, doc in enumerate(docs):
                text = doc.page_content.replace("\n", " ")
                preview = (text[:200] + '...') if len(text) > 200 else text
                meta = doc.metadata.get("source", "N/A")
                sources += f"- Source {i+1} (`{meta}`): {preview}\n"

            message_placeholder.markdown(final_answer + sources)
            full_content = final_answer + sources

        except Exception as e:
            error_msg = f"An error occurred: {e}\nPlease try again."
            message_placeholder.error(error_msg)
            traceback.print_exc()
            full_content = error_msg

    st.session_state.messages.append({"role": "assistant", "content": full_content})

st.markdown("---")
st.markdown("Disclaimer: This bot provides general health info and does not replace professional medical advice.")
