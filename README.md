# Semantic Search API

A self-hostable semantic search API for products using llama.cpp and pgvector. Features distributed tracing, metrics, and configurable similarity thresholds.

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

6. **Load your embedding model**
   ```bash
   # Example: Start llama.cpp server with Qwen3-Embedding-0.6B
   # Make sure it's running at the URL specified in .env (default: http://192.168.0.110:8080)
   ```

7. **Generate embeddings for products**
   ```bash
   python embedding.py
   ```
> For initial testing, you may want to try out with only 500 products. As generating vector for 24K product will take some time. Uncomment `df = df.head(500)` in main function of `embedding.py`
8. **Start the API**
   ```bash
   python main.py
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
- `EMBEDDING_API_URL` - Your llama.cpp server endpoint
- `EMBEDDING_MODEL` - Model name (e.g., Qwen3-Embedding-0.6B)
- `EMBEDDING_DIMENSION` - Output dimensions (must match db.sql)
- `SEARCH_RESULT_LIMIT` - Number of results to return (default: 3)
- `SEARCH_SIMILARITY_THRESHOLD` - Minimum similarity score (default: 0.5) 

## Features

- Structured Logging (structlog)
- Distributed Tracing (OpenTelemetry + Jaeger): 
   - View in Jaeger UI (http://localhost:16686):
- Metrics (Prometheus + Grafana)
   - Prometheus Metrics (http://localhost:9090):

## FAQ

**Q1. Why only 1024 dimensions for embedding?**

I'm using HNSW for vector indexing. With `pgvector` we can't use HNSW indexing with more than 2000 dimension. It's a limitation of `postgresql`. Read [this PR](https://github.com/pgvector/pgvector/issues/461) for more details.

**Q2. HNSW vs IVFFlat**

HNSW (Hierarchical Navigable Small World): Constructs a multi-level graph for faster and highly accurate approximate nearest neighbor searches, generally offering better performance than IVFFlat but using more memory.

IVFFlat (Inverted File with Flat Compression): Partitions the vector space into clusters and only searches the most relevant clusters, trading some accuracy for speed.

HNSW is comparatively faster than IVFFlat.

**Q3. Which embedding should I use?**

Visit [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) and choose one according to your hardware.

> Don't forget to update `EMBEDDING_DIMENSION` in `.env` and `vector(N)` in `db.sql`, then run `make db-reset`.

**Q4. Which LLM should I use?**

Any LLM is fine tbh. Run any smaller ones.

**Q5. How can I serve both LLM and embedding model from a single device?**

You can use [llama-swap](https://github.com/mostlygeek/llama-swap).

I run embedding model and LLM separately with `llama.cpp`.