"""
Product embedding generator and database uploader.
Generates embeddings for product descriptions and stores them in a PostgreSQL database.
"""

import os
import logging
from typing import Dict, List, Any
from datetime import datetime

import pandas as pd
from openai import OpenAI
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

# Initialize OpenAI client
def init_openai() -> OpenAI:
    """Initialize OpenAI client with API key from environment."""
    api_key = os.getenv('OPENAI_KEY')
    if not api_key:
        raise ValueError("Please set the OPENAI_KEY environment variable")
    logger.info("OpenAI client initialized")
    return OpenAI(api_key=api_key)

# Database configuration
def get_db_config() -> Dict[str, str]:
    """Get database configuration from environment variables."""
    config = {
        'dbname': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT')
    }
    
    if None in config.values() or "" in config.values():
        raise ValueError("Missing required database environment variables")
    
    logger.info(f"Database configuration loaded for {config['dbname']} at {config['host']}")
    return config

def generate_product_text(row: pd.Series) -> str:
    """
    Generate a combined text representation of product details.
    
    Args:
        row: DataFrame row containing product information
    Returns:
        Combined product text
    """
    parts = []
    for field in ['brand', 'title', 'category', 'description']:
        if pd.notna(row[field]) and row[field]:
            parts.append(str(row[field]))
    return " ".join(parts)

def get_embedding(text: str, client: OpenAI) -> List[float]:
    """
    Generate embedding for given text using OpenAI's API.
    
    Args:
        text: Input text to generate embedding for
        client: OpenAI client instance
    Returns:
        List of embedding values
    """
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        logger.error(f"Failed text: {text[:100]}...")  # Log first 100 chars of failed text
        raise

def process_dataframe(df: pd.DataFrame, client: OpenAI) -> pd.DataFrame:
    """
    Process DataFrame to add product text and embeddings.
    
    Args:
        df: Input DataFrame with product information
        client: OpenAI client instance
    Returns:
        DataFrame with added product_text and embedding columns
    """
    total_rows = len(df)
    logger.info(f"Starting to process {total_rows} products")
    
    # Generate product text
    logger.info("Generating product text descriptions")
    df['product_text'] = df.apply(generate_product_text, axis=1)
    
    # Generate embeddings with progress tracking
    logger.info("Generating embeddings")
    processed = 0
    def get_embedding_with_progress(text: str) -> List[float]:
        nonlocal processed
        embedding = get_embedding(text, client)
        processed += 1
        if processed % 10 == 0:  # Log every 10 items
            logger.info(f"Processed {processed}/{total_rows} embeddings ({(processed/total_rows)*100:.1f}%)")
        return embedding
    
    df['embedding'] = df['product_text'].apply(get_embedding_with_progress)
    logger.info("Completed generating embeddings")
    
    return df

def insert_to_database(df: pd.DataFrame, db_config: Dict[str, str]) -> None:
    """
    Insert products with embeddings into the database.
    
    Args:
        df: DataFrame with products and embeddings
        db_config: Database connection configuration
    """
    total_rows = len(df)
    logger.info(f"Preparing to insert {total_rows} products into database")
    
    try:
        with psycopg2.connect(**db_config) as conn:
            logger.info("Database connection established")
            with conn.cursor() as cur:
                # Prepare data for insertion
                logger.info("Preparing data for insertion")
                data = []
                for _, row in df.iterrows():
                    # Handle price: convert to float if exists, else None
                    price = row.get('price')
                    if pd.isna(price) or price is None:
                        price = None
                    else:
                        try:
                            price = float(price)
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid price value: {price}, setting to None")
                            price = None
                    
                    data.append((
                        row.get('category', ''),
                        row.get('brand', ''),
                        row.get('title', ''),
                        row.get('description', ''),
                        price,
                        row['embedding']
                    ))
                
                # Insert data using execute_values
                logger.info("Starting database insertion")
                insert_query = """
                    INSERT INTO products (category, brand, title, description, price, embedding)
                    VALUES %s
                """
                execute_values(cur, insert_query, data)
                conn.commit()
                logger.info(f"Successfully inserted {len(data)} products into database")
    
    except psycopg2.Error as e:
        logger.error(f"Database error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during database insertion: {str(e)}")
        raise

def main() -> None:
    """Main function to orchestrate the embedding generation and database upload process."""
    start_time = datetime.now()
    logger.info("Starting product embedding generation and database upload process")
    
    try:
        # Initialize OpenAI
        client = init_openai()
        
        # Get database configuration
        db_config = get_db_config()
        
        # Load and process data
        logger.info("Loading JSON data from cameras-s.json")
        df = pd.read_json('cameras-s.json')
        df = df.head(10)  # Take only first 100 rows
        logger.info(f"Loaded {len(df)} products from JSON file")
        
        df = process_dataframe(df, client)
        
        # Upload to database
        insert_to_database(df, db_config)
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Process completed successfully in {duration}")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()