import json
from typing import Dict, List, Optional, Union, Tuple
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import numpy as np
import re
import ollama
from langchain.embeddings import OllamaEmbeddings
import streamlit as st

# Cache the embedding model initialization at module level
@st.cache_resource
def init_embedding_model():
    """Initialize Ollama embeddings model with caching"""
    return OllamaEmbeddings(model="nomic-embed-text")

class PDFProcessor:
    def __init__(self):
        # Use the cached embedding model
        self.embeddings = init_embedding_model()

    def process_pdf(self, pdf_path: str) -> Dict:
        """
        Main function to process PDFs. Handles both text-based and scanned PDFs.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted information and processing results
        """
        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            
            # Try text extraction first
            text, is_text_pdf = self._extract_text_from_pdf(doc)
            
            if is_text_pdf:
                # Process as text-based PDF
                result = self._process_text_based_pdf(text)
            else:
                # Process as scanned PDF
                result = self._process_scanned_pdf(doc)
                
            doc.close()
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "extracted_text": None,
                "processing_method": "failed"
            }

    def _extract_text_from_pdf(self, doc) -> Tuple[str, bool]:
        """
        Extract text from PDF and determine if it's a text-based PDF.
        
        Returns:
            Tuple of (extracted_text, is_text_pdf)
        """
        text = ""
        for page in doc:
            text += page.get_text()
        
        # If we get substantial text, consider it a text-based PDF
        is_text_pdf = len(text.strip()) > 50
        return text, is_text_pdf

    def _process_text_based_pdf(self, text: str) -> Dict:
        """
        Process text-based PDF using text extraction.
        """
        try:
            # Generate embeddings for the text
            text_embedding = self.embeddings.embed_query(text)
            
            return {
                "extracted_text": text,
                "text_embedding": text_embedding,
                "processing_method": "text_based"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "extracted_text": text,
                "processing_method": "text_based_failed"
            }

    def _process_scanned_pdf(self, doc: fitz.Document) -> Dict:
        """
        Process scanned PDF using OCR.
        """
        try:
            # Process first page
            page = doc[0]
            
            # Convert PDF page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            img_data = pix.tobytes()
            img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
            
            # Perform OCR
            ocr_text = pytesseract.image_to_string(img)
            
            # Generate embeddings
            text_embedding = self.embeddings.embed_query(ocr_text)
            
            return {
                "extracted_text": ocr_text,
                "text_embedding": text_embedding,
                "processing_method": "image_based"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "extracted_text": None,
                "processing_method": "image_based_failed"
            }

def example_usage():
    st.title("PDF Text Processor")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file:
        # Initialize processor
        processor = PDFProcessor()
        
        # Save uploaded file to temporary location
        with st.spinner("Processing PDF..."):
            # Create a temporary file
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Process the temporary file
            result = processor.process_pdf("temp.pdf")
            
            # Clean up
            import os
            os.remove("temp.pdf")
        
        # Display results
        if "error" in result:
            st.error("Processing failed")
            st.write(result["error"])
        else:
            st.success("PDF processed successfully")
            st.write("Extraction Method:", result["processing_method"])
            st.text_area("Extracted Text", result["extracted_text"])

if __name__ == "__main__":
    example_usage()