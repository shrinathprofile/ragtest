import ollama
import streamlit as st
from streamlit_feedback import streamlit_feedback
import uuid
import os
from PyPDF2 import PdfReader
from pathlib import Path
import numpy as np
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader

# Initialize Ollama embeddings model
@st.cache_resource
def init_embedding_model():
    return OllamaEmbeddings(model="nomic-embed-text")

# Function to read and process PDFs
def process_pdfs(folder_path):
    documents = []
    
    pdf_dir = Path(folder_path)
    
    for pdf_path in pdf_dir.glob("*.pdf"):
        try:
            # Use PyPDFLoader instead of direct PdfReader
            loader = PyPDFLoader(str(pdf_path))
            pdf_documents = loader.load()
            documents.extend(pdf_documents)
                
        except Exception as e:
            st.error(f"Error processing {pdf_path.name}: {str(e)}")
            continue
    
    return documents

# Function to process documents and create vector store
def create_vectorstore(documents, embedding_model):
    with st.spinner('Processing documents and creating vector store...'):
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150
        )
        
        # Split documents
        document_chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=document_chunks,
            embedding=embedding_model,
            persist_directory="./chroma_db"
        )
        
        return vectorstore

# Initialize session states
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "run_id" not in st.session_state:
    st.session_state["run_id"] = str(uuid.uuid4())

if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

st.title("Ollama Python Chatbot with PDF Knowledge")

# Initialize embedding model
embedding_model = init_embedding_model()

# Sidebar for PDF processing
with st.sidebar:
    st.header("PDF Processing")
    pdf_folder = st.text_input("Enter PDF folder path:")
    
    if st.button("Process PDFs") and pdf_folder:
        if os.path.exists(pdf_folder):
            # Process PDFs
            documents = process_pdfs(pdf_folder)
            if documents:
                # Create vector store
                st.session_state["vectorstore"] = create_vectorstore(documents, embedding_model)
                st.success(f"Processed {len(documents)} documents into vector store!")
            else:
                st.warning("No documents could be extracted from the PDFs.")
        else:
            st.error("Invalid folder path!")

# Add feedback style toggle
feedback_option = "faces" if st.toggle(label="`Thumbs` ⇄ `Faces`", value=False) else "thumbs"

# Display chat messages from history
for i, message in enumerate(st.session_state["messages"]):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            feedback_key = f"history_feedback_{st.session_state['run_id']}_{i}"
            feedback = streamlit_feedback(
                feedback_type=feedback_option,
                optional_text_label="[Optional] Please provide an explanation",
                key=feedback_key,
            )
            if feedback:
                st.write(f"Feedback received: {feedback}")

# Chat input and response
if prompt := st.chat_input("Ask about the PDFs or enter a city name for travel recommendations"):
    st.session_state["run_id"] = str(uuid.uuid4())
    current_message_idx = len(st.session_state["messages"])
    
    # Add user message to history
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        relevant_context = ""
        if st.session_state["vectorstore"] is not None:
            # Search for relevant documents
            search_results = st.session_state["vectorstore"].similarity_search(
                prompt,
                k=3  # Number of relevant chunks to retrieve
            )
            
            # Create context from relevant documents
            relevant_context = "\n\nRelevant information from PDFs:\n"
            for doc in search_results:
                relevant_context += f"\nFrom page {doc.metadata.get('page', 'unknown')}:\n{doc.page_content}\n"
        
        augmented_prompt = prompt + relevant_context
        
        # Get response from Ollama
        response = ollama.chat(
            model='llama3.2:1b',
            messages=[{"role": "user", "content": augmented_prompt}]
        )
        message = response["message"]["content"]
        st.markdown(message)
        st.session_state["messages"].append({"role": "assistant", "content": message})
        
        # Add feedback for new message
        new_feedback_key = f"new_feedback_{st.session_state['run_id']}_{current_message_idx}"
        feedback = streamlit_feedback(
            feedback_type=feedback_option,
            optional_text_label="[Optional] Please provide an explanation",
            key=new_feedback_key,
        )
        if feedback:
            st.write(f"Feedback received: {feedback}")