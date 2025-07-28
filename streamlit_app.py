import os
from dotenv import load_dotenv
import streamlit as st
import traceback

# LangChain Imports
from langchain_huggingface import HuggingFaceEmbeddings # Corrected import for Embeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAI

# No Imports for Re-ranking (removed)


# --- Configuration ---
load_dotenv() # Load environment variables from .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

DB_FAISS_PATH = "vectorstore/db_faiss" # Path to your saved FAISS vector database

# --- Initialization Functions ---

@st.cache_resource # Cache the vector store to avoid reloading on every rerun
def load_faiss_vector_store(db_path):
    """
    Loads the FAISS vector store from the specified path.
    """
    with st.spinner("Loading knowledge base..."):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    st.success("Knowledge base loaded!")
    return db

@st.cache_resource # Cache the LLM to avoid re-initializing on every rerun
def load_gemini_llm(api_key):
    """
    Initializes and returns the Google Gemini LLM via API.
    """
    if not api_key:
        st.error("Google API Key not found. Please set GOOGLE_API_KEY in your .env file.")
        st.stop() # Stop the app if key is missing

    with st.spinner("Initializing AI model..."):
        llm = GoogleGenerativeAI(
            model="models/gemini-1.5-flash", # Use the model you confirmed works best
            google_api_key=api_key,
            temperature=0.1, # Keep low for factual Q&A
            max_output_tokens=512 # Adjust based on your expected answer length
        )
    st.success("AI model initialized!")
    return llm

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Local Healthcare Info Bot", page_icon="⚕️", layout="centered")
st.title("⚕️ Local Healthcare Info Bot")
st.markdown("Ask me anything about common healthcare topics from our knowledge base!")

# --- Initialize RAG Chain with Memory ---
@st.cache_resource # Cache the chain to avoid re-initializing
def setup_qa_chain():
    vector_db = load_faiss_vector_store(DB_FAISS_PATH)
    llm = load_gemini_llm(GOOGLE_API_KEY)
    
    # Setup memory for ConversationalRetrievalChain
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer" # Explicitly tell memory which key contains the generated answer
    )

    # Define the RAG prompt template for ConversationalRetrievalChain
    qa_template = """Use the following pieces of context to answer the user's question.
If you don't know the answer based on the provided context, politely state that you don't have enough information from the given knowledge base.
Do not make up an answer.
This information is for general knowledge and should not replace professional medical advice.

Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["chat_history", "context", "question"], template=qa_template)

    # ConversationalRetrievalChain is designed for chat history with retrieval
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        # Revert to direct retriever as re-ranking was skipped
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}), # <--- REVERTED: Use base retriever
        memory=memory, # Pass the memory object here
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}, # Apply the prompt to the combine_docs part
        return_source_documents=True, # Ensure source documents are returned
    )
    return qa_chain

# Initialize the QA chain (this will run once due to @st.cache_resource)
qa_chain = setup_qa_chain()

# --- Clear Chat History Button ---
if st.sidebar.button("Clear Chat History"): # Added to sidebar for better layout
    st.session_state.messages = [] # Clear displayed messages
    qa_chain.memory.clear() # Clear LangChain's internal memory buffer
    st.rerun() # Rerun the app to refresh the chat

# --- Chat Interface ---

# Initialize chat history in session state if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me a question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response_text = [] # To accumulate streaming parts
        
        try:
            # Step 1: Stream the answer text
            # Use .stream() method for streaming output. It yields dicts with "answer" key chunks.
            for chunk in qa_chain.stream({"question": prompt}):
                if "answer" in chunk:
                    full_response_text.append(chunk["answer"])
                    message_placeholder.markdown("".join(full_response_text) + "▌") # Add blinking cursor

            # Remove the blinking cursor after streaming is complete
            message_placeholder.markdown("".join(full_response_text))
            
            # Step 2: Get and append sources after streaming (requires another chain invocation for sources)
            # This is less efficient, but ensures sources are displayed correctly with streaming text.
            # A more advanced way involves LangChain callbacks to get intermediate steps during streaming.
            # For simplicity, we'll run a quick invoke for sources only.
            
            # Note: Directly getting sources from `stream()` output is tricky. 
            # The `stream` method yields text fragments. Sources are typically part of the final chain output.
            # To get sources easily after streaming, we can do a quick non-streaming retrieval on the question
            # or rely on the chain's overall execution state if complex callbacks are to be avoided.

            # Re-running the chain with invoke to get sources is inefficient but guaranteed to work.
            # A better way for sources during streaming is to use a callback handler.
            # For this example, let's just make the simple `invoke` call for sources after stream.
            
            # To avoid re-invoking the entire chain, we can fetch sources directly from the retriever if needed,
            # but that won't show *which* sources the LLM *actually used* after prompt stuffing.
            # Let's rely on the final output structure of `invoke` for simplicity to get both.
            # The issue is, you can't get sources *from* the `stream` output directly reliably this way.
            # Let's stick with the invoke() only and forgo streaming for simplicity to ensure sources.
            
            # My previous attempt to combine streaming and sources simultaneously in simple code proved tricky.
            # Given the request to implement streaming, let's do ONLY streaming for the answer
            # and explicitly state that sources with streaming need a more advanced callback handler.
            
            # Let's just stream the answer, and then have a simplified source display after the answer.
            # The user wants "streaming responses". So, let's deliver that.
            
            # --- REVISED STREAMING LOGIC ---
            
            # The full response will be built from streamed chunks
            final_answer_text = "".join(full_response_text)

            # Get sources. ConversationalRetrievalChain doesn't easily expose sources via stream directly.
            # To get sources, we'd either need to use a non-streaming invocation OR
            # set up complex LangChain callbacks. For simplicity, we'll get relevant documents
            # from the retriever directly based on the *current* prompt. This might not be *exactly*
            # what the LLM saw after history + context, but is a reasonable proxy for demo.
            
            retrieved_docs = qa_chain.retriever.get_relevant_documents(prompt)
            
            source_info = ""
            if retrieved_docs:
                source_info += "\n\n**Sources Used:**\n"
                for i, doc in enumerate(retrieved_docs):
                    source_content = doc.page_content.replace("\n", " ") # Remove newlines
                    source_content = (source_content[:200] + '...') if len(source_content) > 200 else source_content
                    source_metadata = doc.metadata.get("source", "N/A")
                    source_info += f"- Source {i+1} (`{source_metadata}`): {source_content}\n"
            
            # Update the message_placeholder with the final answer + sources
            message_placeholder.markdown(final_answer_text + source_info)

            # The full content to store in chat history (for re-display)
            full_content_for_history = final_answer_text + source_info

        except Exception as e:
            error_message = f"An error occurred: {e}\n\nPlease try rephrasing your question or check the API key and internet connection."
            message_placeholder.error(error_message)
            traceback.print_exc() # Print full traceback to console for debugging
            full_content_for_history = error_message # Store error message in history if an error occurred


    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_content_for_history})

st.markdown("---")
st.markdown("Disclaimer: This bot provides general health information and should not replace professional medical advice.")