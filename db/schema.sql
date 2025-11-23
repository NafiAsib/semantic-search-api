-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create product table with embedding vector
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    brand TEXT,
    name TEXT,
    description TEXT,
    price DECIMAL(10, 2),
    image_url TEXT,
    embedding vector(1024),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_products_brand ON products(brand);
CREATE INDEX idx_products_price ON products(price);

-- HNSW index for vector similarity search
CREATE INDEX idx_products_embedding_hnsw ON products USING hnsw (embedding vector_cosine_ops);