"""
API endpoints for product similarity search using embeddings.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
from openai import OpenAI

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Product Search API")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_KEY'))

# Database configuration
DB_CONFIG = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

class SearchQuery(BaseModel):
    query: str

class SearchResult(BaseModel):
    category: str | None = None
    brand: str | None = None
    title: str | None = None
    description: str | None = None
    price: float | None = None
    similarity_score: float

def get_embedding(text: str) -> List[float]:
    """Generate embedding for the search query."""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate embedding")

def search_similar_products(embedding: List[float], limit: int = 3) -> List[Dict[str, Any]]:
    """
    Search for similar products using cosine similarity.
    Returns top matches ordered by similarity score.
    """
    try:
        with psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor) as conn:
            with conn.cursor() as cur:
                # Perform similarity search using dot product
                query = """
                    SELECT 
                        category,
                        brand,
                        title,
                        description,
                        price,
                        1 - (embedding <=> %s::vector) as similarity_score
                    FROM products
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """
                cur.execute(query, (embedding, embedding, limit))
                results = cur.fetchall()
                return [dict(r) for r in results]
    except Exception as e:
        logger.error(f"Database error during search: {str(e)}")
        raise HTTPException(status_code=500, detail="Database search failed")

@app.post("/search", response_model=List[SearchResult])
async def search_products(query: SearchQuery):
    """
    Search for products similar to the query text.
    Returns top 3 most similar products.
    """
    try:
        logger.info(f"Received search query: {query.query}")
        
        # Generate embedding for search query
        query_embedding = get_embedding(query.query)
        logger.info("Generated embedding for search query")
        
        # Search for similar products
        results = search_similar_products(query_embedding)
        logger.info(f"Found {len(results)} similar products")
        
        return results
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing search request: {str(e)}")
        raise HTTPException(status_code=500, detail="Search request failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 