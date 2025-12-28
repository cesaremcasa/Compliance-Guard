import os
import logging
import re
import fitz 
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplianceSmartIngestor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.control_id_pattern = re.compile(r"Control ([A-Z]{2}-\d+(?:\s*\(\d+\))?)")

    def extract_text_from_pdf(self) -> str:
        logger.info("Reading PDF with PyMuPDF...")
        doc = fitz.open(self.pdf_path)
        full_text = []
        
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 50:
                full_text.append(text)
            
            if (page_num + 1) % 50 == 0:
                logger.info(f"Processed {page_num + 1}/{len(doc)} pages...")
                
        doc.close()
        combined_text = "\n\n".join(full_text)
        logger.info(f"Total extracted text length: {len(combined_text)} characters.")
        return combined_text

    def smart_chunk_and_tag(self, text: str) -> List[Document]:
        logger.info("Splitting text into chunks...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " "]
        )
        
        chunks = text_splitter.split_text(text)
        logger.info(f"Created {len(chunks)} chunks.")
        
        documents = []
        current_control_id = "UNKNOWN"
        
        for i, chunk in enumerate(chunks):
            match = self.control_id_pattern.search(chunk[:150]) 
            if match:
                current_control_id = match.group(1)
            
            documents.append(Document(
                page_content=chunk,
                metadata={
                    'control_id': current_control_id,
                    'source': os.path.basename(self.pdf_path),
                    'chunk_index': i
                }
            ))
            
        return documents

    def run_pipeline(self, output_path: str):
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF not found at {self.pdf_path}")

        raw_text = self.extract_text_from_pdf()
        documents = self.smart_chunk_and_tag(raw_text)
        
        logger.info("Generating FAISS Index (This may take a few minutes)...")
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model
        )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        vector_store.save_local(output_path)
        logger.info(f"✅ Ingestion Complete. Index saved to {output_path}")
        
        return True

if __name__ == "__main__":
    ingestor = ComplianceSmartIngestor("data/official_nist.pdf")
    ingestor.run_pipeline("data/processed/faiss_index.bin")
