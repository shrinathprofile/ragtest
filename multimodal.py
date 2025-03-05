import streamlit as st
import ollama
from streamlit_feedback import streamlit_feedback
import uuid
import os
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import Document
import fitz  # PyMuPDF
import io
from pdfplumber import open as pdf_open
import cv2
import tempfile
import re  # Added for reference extraction

st.set_page_config(
    page_title="PDF Multimodal RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def init_embedding_model():
    return OllamaEmbeddings(model="nomic-embed-text")

def extract_images_from_pdf(pdf_path, original_filename):
    documents = []
    image_descriptions = []
    pdf_document = fitz.open(pdf_path)
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                
                image = Image.open(io.BytesIO(image_bytes))
                st.image(image, caption=f"Page {page_num + 1}, Image {img_index}", use_column_width=True)
                
                response = ollama.chat(
                    model='llava:7b',
                    messages=[{
                        'role': 'user',
                        'content': 'Describe this image in detail, including any visible text or numbers.',
                        'images': [image_bytes]
                    }]
                )
                
                image_description = response["message"]["content"]
                if "sorry" in image_description.lower() or "no image" in image_description.lower():
                    image_description = f"Image extraction failed for page {page_num + 1}, image {img_index}. Style suggests a presentation slide or chart."
                    st.warning(f"Image description failed for {original_filename}, page {page_num + 1}")
                
                image_descriptions.append({
                    "page": page_num + 1,
                    "image_index": img_index,
                    "description": image_description
                })
                
                doc = Document(
                    page_content=image_description,
                    metadata={
                        "source": original_filename,
                        "page": page_num + 1,
                        "type": "image",
                        "image_index": img_index
                    }
                )
                documents.append(doc)
                
            except Exception as e:
                st.warning(f"Error processing image {img_index} on page {page_num + 1}: {str(e)}")
                continue
    
    pdf_document.close()
    return documents, image_descriptions

def extract_tables_from_pdf(pdf_path, original_filename):
    documents = []
    
    try:
        with pdf_open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                
                for table_idx, table in enumerate(tables):
                    if table:
                        table_str = "\n".join([" | ".join([str(cell) for cell in row if cell]) for row in table if any(row)])
                        doc = Document(
                            page_content=f"Table content:\n{table_str}",
                            metadata={
                                "source": original_filename,
                                "type": "table",
                                "page": page_num + 1,
                                "table_index": table_idx
                            }
                        )
                        documents.append(doc)
                        
                        headers = table[0] if table else []
                        for row_idx, row in enumerate(table[1:], 1):
                            row_content = " | ".join([
                                f"{headers[i] if i < len(headers) else f'Column {i}'}: {cell}" 
                                for i, cell in enumerate(row) 
                                if cell and cell.strip()
                            ])
                            if row_content.strip():
                                doc = Document(
                                    page_content=row_content,
                                    metadata={
                                        "source": original_filename,
                                        "type": "table_row",
                                        "page": page_num + 1,
                                        "table_index": table_idx,
                                        "row_index": row_idx
                                    }
                                )
                                documents.append(doc)
    
    except Exception as e:
        st.error(f"Error extracting tables: {str(e)}")
    
    return documents

def extract_text_from_pdf(pdf_path, original_filename):
    documents = []
    pdf_document = fitz.open(pdf_path)
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        text = page.get_text()
        
        if text.strip():
            doc = Document(
                page_content=text,
                metadata={
                    "source": original_filename,
                    "page": page_num + 1,
                    "type": "text"
                }
            )
            documents.append(doc)
    
    pdf_document.close()
    return documents

def process_pdf(pdf_path, original_filename):
    documents = []
    image_descriptions = []
    
    with st.spinner(f'Processing {original_filename}...'):
        text_docs = extract_text_from_pdf(pdf_path, original_filename)
        documents.extend(text_docs)
        st.write(f"- Extracted {len(text_docs)} text segments")
        
        table_docs = extract_tables_from_pdf(pdf_path, original_filename)
        documents.extend(table_docs)
        st.write(f"- Extracted {len(table_docs)} table elements")
        
        image_docs, image_descriptions = extract_images_from_pdf(pdf_path, original_filename)
        documents.extend(image_docs)
        st.write(f"- Extracted {len(image_docs)} images")
    
    return documents, image_descriptions

def create_vectorstore(documents, embedding_model):
    with st.spinner('Creating vector store...'):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        
        document_chunks = text_splitter.split_documents(documents)
        
        vectorstore = Chroma.from_documents(
            documents=document_chunks,
            embedding=embedding_model,
            persist_directory="./chroma_db"
        )
        
        return vectorstore

def render_pdf_with_highlight(pdf_path, page_num, content, content_type):
    try:
        pdf_document = fitz.open(pdf_path)
        page = pdf_document[page_num - 1]
        
        if content_type == "text":
            text_instances = page.search_for(content[:100])
            if text_instances:
                for inst in text_instances:
                    highlight = page.add_highlight_annot(inst)
                    highlight.update()
        
        elif content_type == "table":
            text_instances = page.search_for(content.split("\n")[0][:50])
            if text_instances:
                for inst in text_instances:
                    highlight = page.add_highlight_annot(inst)
                    highlight.update()
        
        elif content_type == "image":
            image_list = page.get_images()
            for img in image_list:
                rect = page.get_image_bbox(img)
                highlight = page.add_highlight_annot(rect)
                highlight.update()

        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        pdf_document.close()
        return img
    
    except Exception as e:
        st.error(f"Error rendering PDF: {str(e)}")
        return None

# Function to extract references from response
def extract_references(text):
    pattern = r'\[REF(\d+)\]'
    matches = re.findall(pattern, text)
    return [int(m) for m in matches]

# Initialize session states
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "run_id" not in st.session_state:
    st.session_state["run_id"] = str(uuid.uuid4())
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "image_descriptions" not in st.session_state:
    st.session_state["image_descriptions"] = []
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []
if "selected_source" not in st.session_state:
    st.session_state["selected_source"] = None
if "references" not in st.session_state:
    st.session_state["references"] = {}  # Added to store references

st.title("PDF Multimodal RAG System")
st.markdown("Upload PDFs and ask questions about their text, tables, and images!")

embedding_model = init_embedding_model()

# Two-column layout
left_column, right_column = st.columns([2, 1])

show_debug = st.sidebar.checkbox("Show Debug Info", value=False)

with st.sidebar:
    st.header("PDF Processing")
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if st.button("Process PDFs") and uploaded_files:
        st.session_state["uploaded_files"] = uploaded_files
        all_documents = []
        all_image_descriptions = []
        
        if st.session_state["vectorstore"] is not None:
            st.session_state["vectorstore"].delete_collection()
            st.session_state["vectorstore"] = None
            st.write("Cleared previous vector store.")
        
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            docs, image_descriptions = process_pdf(tmp_path, uploaded_file.name)
            all_documents.extend(docs)
            all_image_descriptions.extend(image_descriptions)
            
            os.unlink(tmp_path)
        
        if all_documents:
            st.session_state["vectorstore"] = create_vectorstore(all_documents, embedding_model)
            st.success(f"Created vector store with {len(all_documents)} total elements!")
        else:
            st.warning("No content could be extracted from the PDFs.")
        
        st.session_state["image_descriptions"] = all_image_descriptions

    if st.session_state["image_descriptions"]:
        st.header("Image Descriptions")
        for desc in st.session_state["image_descriptions"]:
            st.write(f"**Page {desc['page']}, Image {desc['image_index']}:**")
            st.write(desc["description"])
            st.write("---")

# Left column: Chat interface
with left_column:
    feedback_option = "faces" if st.toggle(label="`Thumbs` ⇄ `Faces`", value=False) else "thumbs"

    for i, message in enumerate(st.session_state["messages"]):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display reference buttons for assistant messages
            if message["role"] == "assistant" and i in st.session_state["references"]:
                references = st.session_state["references"][i]
                if references:
                    st.write("**References:**")
                    for ref_num, ref_data in sorted(references.items()):
                        button_label = f"[REF{ref_num}] Page {ref_data['page']}"
                        if st.button(button_label, key=f"ref_btn_{i}_{ref_num}"):
                            st.session_state["selected_source"] = {
                                "file": ref_data["source"],
                                "page": ref_data["page"],
                                "content": ref_data["content"],
                                "type": ref_data["type"]
                            }
                            st.rerun()

            if message["role"] == "assistant":
                feedback_key = f"feedback_{st.session_state['run_id']}_{i}"
                feedback = streamlit_feedback(
                    feedback_type=feedback_option,
                    optional_text_label="[Optional] Please provide an explanation",
                    key=feedback_key,
                )
                if feedback:
                    st.write(f"Feedback received: {feedback}")

    if prompt := st.chat_input("Ask about your PDFs..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            current_message_idx = len(st.session_state["messages"])
            relevant_context = ""
            retrieved_docs = []
            references = {}

            if st.session_state["vectorstore"] is not None:
                search_results = st.session_state["vectorstore"].similarity_search(
                    prompt,
                    k=5
                )
                retrieved_docs = search_results
                
                if show_debug:
                    with st.expander("Retrieved Documents for Debugging"):
                        for doc in retrieved_docs:
                            st.write(f"Source: {doc.metadata['source']}, Page: {doc.metadata.get('page', 'N/A')}, Type: {doc.metadata['type']}")
                            st.write(f"Content: {doc.page_content[:200]}...")
                
                relevant_context = "\n\nRelevant information from documents:\n"
                for idx, doc in enumerate(search_results):
                    ref_num = idx + 1
                    source_type = doc.metadata.get('type', 'document')
                    source_name = doc.metadata.get('source', 'unknown')
                    page_num = doc.metadata.get('page', None)
                    
                    source_info = f"From {source_type} in '{source_name}'"
                    if page_num:
                        source_info += f" (page {page_num})"
                    
                    relevant_context += f"\n[REF{ref_num}] {source_info}:\n{doc.page_content}\n"
                    
                    # Store reference data
                    references[ref_num] = {
                        "source": source_name,
                        "page": page_num,
                        "type": source_type,
                        "content": doc.page_content
                    }
            
            augmented_prompt = f"""Based on the following context extracted from PDF documents, determine the main topic of the PDF and provide a concise response to the query. The context includes text, table data, and image descriptions (if available). For each piece of information you use, include a reference code like [REF1], [REF2], etc., corresponding to the order of the context items provided. Use only the provided information—do not assume additional data. Summarize the most relevant pieces of extracted content and cite each source in your response.

            Query: {prompt}

            {relevant_context}"""
            
            response = ollama.chat(
                model='qwen2.5:3b',
                messages=[{"role": "user", "content": augmented_prompt}]
            )
            message = response["message"]["content"]
            
            # Extract and clean references
            ref_numbers = extract_references(message)
            response_cleaned = re.sub(r'\[REF\d+\]', '', message).strip()
            
            st.markdown(response_cleaned)
            
            # Display reference buttons
            if ref_numbers:
                st.write("**References:**")
                for ref_num in sorted(set(ref_numbers)):
                    if ref_num in references:
                        ref_data = references[ref_num]
                        button_label = f"[REF{ref_num}] Page {ref_data['page']}"
                        if st.button(button_label, key=f"ref_btn_{current_message_idx}_{ref_num}"):
                            st.session_state["selected_source"] = {
                                "file": ref_data["source"],
                                "page": ref_data["page"],
                                "content": ref_data["content"],
                                "type": ref_data["type"]
                            }
                            st.rerun()
            
            # Store message and references
            st.session_state["messages"].append({"role": "assistant", "content": response_cleaned})
            st.session_state["references"][current_message_idx] = references
            
            new_feedback_key = f"new_feedback_{st.session_state['run_id']}_{current_message_idx}"
            feedback = streamlit_feedback(
                feedback_type=feedback_option,
                optional_text_label="[Optional] Please provide an explanation",
                key=new_feedback_key,
            )
            if feedback:
                st.write(f"Feedback received: {feedback}")

# Right column: PDF viewer
with right_column:
    with st.expander("PDF Viewer", expanded=False):
        if st.session_state["uploaded_files"]:
            # Default to first uploaded file, page 1 if no source selected
            pdf_to_show = st.session_state["uploaded_files"][0]
            page_num = 1
            content = ""
            content_type = "text"
            
            # Update based on selected source
            if "selected_source" in st.session_state and st.session_state["selected_source"]:
                selected_source = st.session_state["selected_source"]
                pdf_to_show = next((f for f in st.session_state["uploaded_files"] if f.name == selected_source["file"]), pdf_to_show)
                page_num = selected_source["page"]
                content = selected_source["content"]
                content_type = selected_source["type"]
                st.write(f"Rendering: {selected_source['file']}, Page {page_num}, Type: {content_type}")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_to_show.getvalue())
                pdf_path = tmp_file.name
            
            highlighted_img = render_pdf_with_highlight(pdf_path, page_num, content, content_type)
            if highlighted_img:
                st.image(highlighted_img, caption=f"Page {page_num} of {pdf_to_show.name}", use_column_width=True)
            else:
                # Fallback to unhighlighted page
                pdf_doc = fitz.open(pdf_path)
                page = pdf_doc[page_num - 1]
                pix = page.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                st.image(img, caption=f"Page {page_num} of {pdf_to_show.name} (No highlight)", use_column_width=True)
                pdf_doc.close()
            
            os.unlink(pdf_path)
        else:
            st.write("No PDF uploaded yet.")