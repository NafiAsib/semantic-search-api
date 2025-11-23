# Semantic Search API

A self-hostable search API to search products using vector embeddings. 

Features two-stage retrieval (semantic search + reranking), distributed tracing, metrics, and configurable similarity thresholds.

- [Dataset link](https://www.kaggle.com/datasets/nafiasib/amazon-best-sellers-product-dataset)

## Quick Start

1. **Clone the repository**
   ```bash
   git clone git@github.com:NafiAsib/semantic-search-api.git
   cd semantic-search-api
   ```
2.  **Dataset**

- Download the dataset into `/dataset` directory

```bash
curl -L -o $(pwd)/dataset/amazon-best-sellers-product-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/nafiasib/amazon-best-sellers-product-dataset
```

- Unzip & remove zip file

```bash
unzip dataset/amazon-best-sellers-product-dataset.zip -d dataset/
rm dataset/amazon-best-sellers-product-dataset.zip
```

2. **Configure environment**
   ```bash
   cp .env.example .env # Edit .env with your settings
   ```

3. **Start services with Docker Compose**
   ```bash
   make docker-up
   ```
   This starts:
   - PostgreSQL with pgvector extension
   - Jaeger for distributed tracing
   - Prometheus for metrics
   - Grafana for visualization

4. **Create database schema**
   ```bash
   make db-reset
   ```

5. **Install Python dependencies**
   ```bash
   uv sync
   ```

6. **Load your embedding and reranker models**
   ```bash
   # Start llama.cpp server with Qwen3-Embedding-0.6B on port 8080
   # Make sure it's running at the URL specified in .env (default: http://192.168.0.110:8080)
   
   # Start llama.cpp server with Qwen3-Reranker-0.6B on port 8081
   # Make sure it's running at the URL specified in .env (default: http://192.168.0.110:8081)
   ```

7. **Generate embeddings for products**
   ```bash
   python app/embedding.py
   ```
> For initial testing, you may want to try out with only 500 products. As generating vector for 24K product will take some time. Uncomment `df = df.head(500)` in main function of `app/embedding.py`
8. **Start the API**
   ```bash
   python app/main.py
   ```

9. **Test the search endpoint**
   ```bash
   curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "wireless headphones"}'
   ```

## Makefile Commands

- `make docker-up` - Start all Docker services
- `make docker-down` - Stop all Docker services
- `make db-create` - Create database schema and extensions
- `make db-reset` - Drop and recreate database (destroys data)
- `make db-status` - Show database status and table info
- `make db-connect` - Connect to database via psql
- `make help` - Show all available commands

## Services

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Metrics**: http://localhost:8000/metrics
- **Grafana**: http://localhost:3000
- **Prometheus**: http://localhost:9090
- **Jaeger UI**: http://localhost:16686
- **PostgreSQL**: localhost:5432

## Configuration

Key environment variables in `.env`:

**Embedding Configuration:**
- `EMBEDDING_API_URL` - Your llama.cpp server endpoint (default: http://192.168.0.110:8080)
- `EMBEDDING_MODEL` - Model name (e.g., Qwen3-Embedding-0.6B)
- `EMBEDDING_DIMENSION` - Output dimensions (must match db.sql)

**Reranker Configuration:**
- `RERANKER_API_URL` - Your llama.cpp reranker endpoint (default: http://192.168.0.110:8081)
- `RERANKER_MODEL` - Reranker model name (default: Qwen/Qwen3-Reranker-0.6B)
- `RERANKER_CANDIDATES` - Number of candidates to fetch for reranking (default: 20)

**Search Configuration:**
- `SEARCH_RESULT_LIMIT` - Number of results to return (default: 3)
- `SEARCH_SIMILARITY_THRESHOLD` - Minimum similarity score (default: 0.5) 

## Features

- **Two-Stage Retrieval**: Semantic search with vector similarity + reranking for improved relevance
- **Structured Logging** (structlog)
- **Distributed Tracing** (OpenTelemetry + Jaeger): 
   - View in Jaeger UI (http://localhost:16686)
- **Metrics** (Prometheus + Grafana)
   - Prometheus Metrics (http://localhost:9090)

## FAQ

**Q1. Why only 1024 dimensions for embedding?**

I'm using HNSW for vector indexing. With `pgvector` we can't use HNSW indexing with more than 2000 dimension. It's a limitation of `postgresql`. Read [this PR](https://github.com/pgvector/pgvector/issues/461) for more details.

**Q2. HNSW vs IVFFlat**

HNSW (Hierarchical Navigable Small World): Constructs a multi-level graph for faster and highly accurate approximate nearest neighbor searches, generally offering better performance than IVFFlat but using more memory.

IVFFlat (Inverted File with Flat Compression): Partitions the vector space into clusters and only searches the most relevant clusters, trading some accuracy for speed.

HNSW is comparatively faster than IVFFlat.

**Q3. Which embedding should I use?**

Visit [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) and choose one according to your hardware.

> Don't forget to update `EMBEDDING_DIMENSION` in `.env` and `vector(N)` in `db/schema.sql`, then run `make db-reset`.

**Q4. Why use a reranker?**

Reranking significantly improves search quality. The two-stage approach:
1. **Semantic Search**: Fast vector similarity retrieves 20 candidates
2. **Reranking**: Qwen3-Reranker-0.6B analyzes query-document pairs for better relevance scoring

This gives you speed (from vector search) + accuracy (from reranking).

**Q5. Which reranker should I use?**

Qwen3-Reranker-0.6B is a great lightweight option. For other choices, check the [MTEB Retrieval Leaderboard](https://huggingface.co/spaces/mteb/leaderboard). Note: llama.cpp only supports models with reranking capability.

**Q6. Do I need separate servers for embedding and reranking?**

Yes, since llama.cpp runs one model per server. Run them on different ports (8080 for embedding, 8081 for reranker). You can use [llama-swap](https://github.com/mostlygeek/llama-swap) if you want to optimize memory by swapping models dynamically.