"""
Product embedding generator and database uploader.
Generates embeddings for product descriptions and stores them in a PostgreSQL database.
"""

import os
import logging
from typing import Dict, List
from datetime import datetime

import pandas as pd
import requests
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()
logger.info("Environment variables loaded")

EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://192.168.0.110:8080")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen3-Embedding-8B")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "4096"))


def get_db_config() -> Dict[str, str]:
    """Get database configuration from environment variables."""
    config = {
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
    }

    if None in config.values() or "" in config.values():
        raise ValueError("Missing required database environment variables")

    logger.info(
        f"Database configuration loaded for {config['dbname']} at {config['host']}"
    )
    return config


def generate_product_text(row: pd.Series) -> str:
    """
    Generate a combined text representation of product details for embedding.

    Args:
        row: DataFrame row containing product information
    Returns:
        Combined product text
    """
    parts = []

    if pd.notna(row.get("name")) and row.get("name"):
        parts.append(f"Product: {row['name']}")

    if pd.notna(row.get("brandName")) and row.get("brandName"):
        parts.append(f"Brand: {row['brandName']}")

    if pd.notna(row.get("description")) and row.get("description"):
        parts.append(f"Description: {row['description']}")

    if pd.notna(row.get("features")) and row.get("features"):
        parts.append(f"Features: {row['features']}")

    return " ".join(parts)


def get_embedding(text: str) -> List[float]:
    """
    Generate embedding for given text using self-hosted API.

    Args:
        text: Input text to generate embedding for
    Returns:
        List of embedding values
    """
    # Truncate text if too long (keep reasonable length for context)
    max_chars = 8000  # Adjust based on your model's context length
    if len(text) > max_chars:
        logger.warning(f"Text too long ({len(text)} chars), truncating to {max_chars}")
        text = text[:max_chars]

    try:
        response = requests.post(
            f"{EMBEDDING_API_URL}/v1/embeddings",
            json={"model": EMBEDDING_MODEL, "input": text, "encoding_format": "float"},
            headers={"Content-Type": "application/json"},
            timeout=60,
        )

        # Log detailed error information
        if response.status_code != 200:
            logger.error(f"API returned status {response.status_code}")
            logger.error(f"Response body: {response.text}")
            logger.error(f"Request text length: {len(text)}")
            logger.error(f"First 200 chars: {text[:200]}")

        response.raise_for_status()
        data = response.json()
        embedding = data["data"][0]["embedding"]

        # Validate embedding dimension
        if len(embedding) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected embedding dimension {EMBEDDING_DIMENSION}, got {len(embedding)}"
            )

        return embedding
    except requests.exceptions.RequestException as e:
        logger.error(f"Error generating embedding: {str(e)}")
        logger.error(f"Failed text (first 200 chars): {text[:200]}...")
        raise
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        logger.error(f"Failed text (first 200 chars): {text[:200]}...")
        raise


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process DataFrame to add product text and embeddings.

    Args:
        df: Input DataFrame with product information
    Returns:
        DataFrame with added product_text and embedding columns
    """
    total_rows = len(df)
    logger.info(f"Starting to process {total_rows} products")

    # Generate product text descriptions
    logger.info("Generating product text descriptions")
    df["product_text"] = df.apply(generate_product_text, axis=1)

    # Generate embeddings
    logger.info("Generating embeddings")
    embeddings = []
    processed = 0
    failed = 0

    for idx, text in enumerate(df["product_text"]):
        try:
            embedding = get_embedding(text)
            embeddings.append(embedding)
            processed += 1
            if processed % 10 == 0 or processed == total_rows:
                logger.info(
                    f"Processed {processed}/{total_rows} embeddings ({(processed / total_rows) * 100:.1f}%), failed: {failed}"
                )
        except Exception as e:
            logger.error(f"Failed to generate embedding for row {idx}, skipping...")
            embeddings.append(None)  # Mark as None to filter out later
            failed += 1

    df["embedding"] = embeddings

    # Filter out failed embeddings
    original_count = len(df)
    df = df[df["embedding"].notna()].reset_index(drop=True)
    filtered_count = original_count - len(df)

    if filtered_count > 0:
        logger.warning(
            f"Filtered out {filtered_count} products due to embedding failures"
        )

    logger.info(
        f"Completed generating embeddings: {len(df)} successful, {failed} failed"
    )

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
                    # Extract brand
                    brand = row.get("brandName", "")
                    if pd.isna(brand):
                        brand = ""

                    # Extract name
                    name = row.get("name", "")
                    if pd.isna(name):
                        name = ""

                    # Extract description
                    description = row.get("description", "")
                    if pd.isna(description):
                        description = ""

                    # Handle price: convert to float if exists, else None
                    price = row.get("salePrice")
                    if pd.isna(price) or price is None or price == "":
                        price = None
                    else:
                        try:
                            price = float(price)
                        except (ValueError, TypeError):
                            logger.warning(
                                f"Invalid price value: {price}, setting to None"
                            )
                            price = None

                    # Extract image URL (take first image from imageUrls)
                    image_url = ""
                    image_urls = row.get("imageUrls", "")
                    if pd.notna(image_urls) and image_urls:
                        # Split by comma and take first URL
                        urls = str(image_urls).split(",")
                        if urls:
                            image_url = urls[0].strip()

                    data.append(
                        (brand, name, description, price, image_url, row["embedding"])
                    )

                # Insert data using execute_values
                logger.info("Starting database insertion")
                insert_query = """
                    INSERT INTO products (brand, name, description, price, image_url, embedding)
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
        # Get database configuration
        db_config = get_db_config()

        # Load and process data from CSV
        csv_path = "dataset/products.csv"
        logger.info(f"Loading CSV data from {csv_path}")

        # Read CSV with proper handling
        df = pd.read_csv(csv_path, low_memory=False)
        logger.info(f"Loaded {len(df)} products from CSV file")

        # df = df.head(500)  # for testing
        logger.info(f"Processing first {len(df)} products for testing")

        df = process_dataframe(df)

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
