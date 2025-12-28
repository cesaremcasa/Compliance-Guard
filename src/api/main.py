from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging
import sys
import os

# Add parent directory to path to import rag modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.rag.retriever import retriever_instance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Compliance-Guard API",
    description="API for retrieving compliance controls using RAG.",
    version="0.1.0"
)

# --- Request/Response Models ---
class QueryRequest(BaseModel):
    query: str = Field(..., description="The natural language query about compliance standards.", min_length=3)
    top_k: int = Field(default=3, ge=1, le=10, description="Number of results to return.")

class ControlResult(BaseModel):
    control_id: str
    source: str
    content: str

class QueryResponse(BaseModel):
    query: str
    results: list[ControlResult]

# --- Endpoints ---

@app.get("/")
def health_check():
    """Health check endpoint to verify the API is running."""
    return {"status": "healthy", "service": "Compliance-Guard v3.0"}

@app.post("/search", response_model=QueryResponse)
def search_controls(request: QueryRequest):
    """
    Search the compliance knowledge base for relevant controls.
    
    Example:
    - Query: "How do I manage user accounts?"
    - Returns: A list of controls like AC-2 with context.
    """
    try:
        # Delegate to the retriever module
        raw_results = retriever_instance.search(request.query, k=request.top_k)
        
        # Convert raw dicts to Pydantic models for validation
        formatted_results = [ControlResult(**item) for item in raw_results]
        
        return QueryResponse(query=request.query, results=formatted_results)
    
    except FileNotFoundError as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail="Knowledge base index not found.")
    except Exception as e:
        logger.error(f"Internal server error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during search.")

if __name__ == "__main__":
    import uvicorn
    # Note: For production, bind to 0.0.0.0
    uvicorn.run(app, host="0.0.0.0", port=8000)
