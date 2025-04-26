-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create product table with embedding vector
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    category VARCHAR(255),
    brand VARCHAR(255),
    title TEXT,
    description TEXT,
    price DECIMAL(10, 2),
    embedding vector(1536), -- Adjust dimension based on your embedding model (1536 for OpenAI, 384 for MiniLM)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_products_category ON products(category);
CREATE INDEX idx_products_brand ON products(brand);
CREATE INDEX idx_products_price ON products(price);

CREATE INDEX idx_products_embedding_hnsw ON products USING hnsw (embedding vector_cosine_ops);