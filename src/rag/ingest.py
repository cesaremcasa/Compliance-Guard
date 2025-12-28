import os
import logging
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# CORREÇÃO FINAL: Document mudou para langchain_core.documents na versão mais recente
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/ingestion.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MockDataGenerator:
    @staticmethod
    def create_mock_pdf(path: str) -> None:
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
        except ImportError:
            logger.error("Library 'reportlab' is missing.")
            raise

        logger.warning(f"No PDF found. Generating mock PDF at {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        try:
            c = canvas.Canvas(path, pagesize=letter)
            width, height = letter
            text_content = [
                "NIST SP 800-53 Rev 5",
                "Control AC-2: Account Management",
                "The organization identifies account types and assigns managers.",
                "Control AC-3: Access Enforcement",
                "The system enforces approved authorizations for logical access."
            ]
            y_position = height - 100
            for line in text_content:
                c.drawString(100, y_position, line)
                y_position -= 30
            c.save()
            logger.info(f"Mock PDF created at {path}")
        except Exception as e:
            logger.error(f"Failed to create PDF: {e}")
            raise

def extract_metadata(page_content: str, page_metadata: Dict[str, Any]) -> Dict[str, Any]:
    import re
    control_pattern = r"(?:Control\s+)?([A-Z]{2}-\d{1,2})"
    matches = re.findall(control_pattern, page_content)
    page_metadata['control_id'] = matches[0] if matches else "UNKNOWN"
    return page_metadata

def run_ingestion_pipeline(data_dir: str = "data/raw", index_path: str = "data/processed/faiss_index.bin", model_name: str = "BAAI/bge-small-en-v1.5"):
    try:
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
        if not pdf_files:
            mock_path = os.path.join(data_dir, "mock_nist_doc.pdf")
            MockDataGenerator.create_mock_pdf(mock_path)
            pdf_files = [os.path.basename(mock_path)]

        documents: List[Document] = []
        for pdf_file in pdf_files:
            file_path = os.path.join(data_dir, pdf_file)
            logger.info(f"Loading: {file_path}")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)

        for doc in documents:
            doc.page_content = doc.page_content.strip()
            doc.metadata = extract_metadata(doc.page_content, doc.metadata)
            doc.metadata['source'] = pdf_file

        logger.info("Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")

        logger.info("Initializing embeddings...")
        embedding_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})

        logger.info("Creating FAISS index...")
        vector_store = FAISS.from_documents(documents=chunks, embedding=embedding_model)
        vector_store.save_local(index_path)
        logger.info(f"Index saved to {index_path}")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    run_ingestion_pipeline()
