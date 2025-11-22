# Semantic Search API

- [Dataset link](https://www.kaggle.com/datasets/piyushkumar509/random-products-and-their-descriptions)

## Quick Start

1. **Clone the repository**
   ```bash
   git clone git@github.com:NafiAsib/semantic-search-api.git
   cd semantic-search-api
   ```

2. **Start services with Docker Compose**
   ```bash
   docker compose up -d
   ```
   This starts:
   - PostgreSQL with pgvector extension
   - Jaeger for distributed tracing
   - Prometheus for metrics
   - Grafana for visualization

3. **Install Python dependencies**
   ```bash
   uv sync
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

## Services

- **API**: http://localhost:8000 (once implemented)
- **Grafana**: http://localhost:3000
- **Prometheus**: http://localhost:9090
- **Jaeger UI**: http://localhost:16686
- **PostgreSQL**: localhost:5432

## Environment Variables

Rename the `.env.example` to `.env` and modify values as per your configuration.