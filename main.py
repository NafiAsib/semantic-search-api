"""
API endpoints for product similarity search using embeddings.
"""

import os
import time
from typing import List, Dict, Any
from datetime import datetime
import json

from dotenv import load_dotenv
import logging
import structlog
import requests
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

load_dotenv()

# Initialize OpenTelemetry
resource = Resource.create({"service.name": "semantrix-api"})
tracer_provider = TracerProvider(resource=resource)
otlp_exporter = OTLPSpanExporter(
    endpoint=os.getenv("OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
)
tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer(__name__)

# Initialize Prometheus metrics
SEARCH_REQUESTS = Counter(
    "search_requests_total", "Total number of search requests", ["status"]
)
SEARCH_LATENCY = Histogram(
    "search_request_latency_seconds",
    "Search request latency in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
)
EMBEDDING_LATENCY = Histogram(
    "embedding_generation_latency_seconds", "Embedding generation latency in seconds"
)
DB_QUERY_LATENCY = Histogram(
    "db_query_latency_seconds", "Database query latency in seconds"
)

# Initialize FastAPI app
app = FastAPI(title="Product Search API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Embedding API configuration
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://192.168.0.110:8080")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen3-Embedding-8B")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "4096"))

# Search configuration
SEARCH_RESULT_LIMIT = int(os.getenv("SEARCH_RESULT_LIMIT", "3"))
SEARCH_SIMILARITY_THRESHOLD = float(os.getenv("SEARCH_SIMILARITY_THRESHOLD", "0.5"))

# Database configuration
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}


class SearchQuery(BaseModel):
    query: str


class SearchResult(BaseModel):
    brand: str | None = None
    name: str | None = None
    description: str | None = None
    price: float | None = None
    image_url: str | None = None
    similarity_score: float


@EMBEDDING_LATENCY.time()
def get_embedding(text: str) -> List[float]:
    """Generate embedding for the search query using self-hosted API."""
    with tracer.start_as_current_span("generate_embedding") as span:
        span.set_attribute("text.length", len(text))
        try:
            response = requests.post(
                f"{EMBEDDING_API_URL}/v1/embeddings",
                json={
                    "model": EMBEDDING_MODEL,
                    "input": text,
                    "encoding_format": "float",
                },
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            embedding = data["data"][0]["embedding"]

            # Validate embedding dimension
            if len(embedding) != EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Expected embedding dimension {EMBEDDING_DIMENSION}, got {len(embedding)}"
                )

            span.set_status(Status(StatusCode.OK))
            return embedding
        except requests.exceptions.RequestException as e:
            span.set_status(Status(StatusCode.ERROR), str(e))
            span.record_exception(e)
            logger.error(
                "embedding_api_request_failed", error=str(e), text_length=len(text)
            )
            raise HTTPException(
                status_code=500, detail=f"Failed to generate embedding: {str(e)}"
            )
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR), str(e))
            span.record_exception(e)
            logger.error(
                "embedding_generation_failed", error=str(e), text_length=len(text)
            )
            raise HTTPException(status_code=500, detail="Failed to generate embedding")


@DB_QUERY_LATENCY.time()
def search_similar_products(
    embedding: List[float], limit: int = None, threshold: float = None
) -> List[Dict[str, Any]]:
    """
    Search for similar products using cosine similarity.
    Returns top matches ordered by similarity score that meet the threshold.
    Returns empty list if no products meet the similarity threshold.
    """
    if limit is None:
        limit = SEARCH_RESULT_LIMIT
    if threshold is None:
        threshold = SEARCH_SIMILARITY_THRESHOLD

    with tracer.start_as_current_span("search_similar_products") as span:
        span.set_attribute("limit", limit)
        span.set_attribute("threshold", threshold)
        try:
            with psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor) as conn:
                with conn.cursor() as cur:
                    query = """
                        SELECT 
                            brand,
                            name,
                            description,
                            price,
                            image_url,
                            1 - (embedding <=> %s::vector) as similarity_score
                        FROM products
                        WHERE 1 - (embedding <=> %s::vector) >= %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s;
                    """
                    cur.execute(
                        query, (embedding, embedding, threshold, embedding, limit)
                    )
                    results = cur.fetchall()
                    span.set_attribute("results.count", len(results))
                    span.set_status(Status(StatusCode.OK))
                    return [dict(r) for r in results]
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR), str(e))
            span.record_exception(e)
            logger.error(
                "database_search_failed", error=str(e), limit=limit, threshold=threshold
            )
            raise HTTPException(status_code=500, detail="Database search failed")


@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(time.time_ns()))
    structlog.contextvars.bind_contextvars(request_id=request_id)

    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = str(duration)
    return response


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")


@app.post("/search", response_model=List[SearchResult])
async def search_products(query: SearchQuery, request: Request):
    """
    Search for products similar to the query text.
    Returns top N most similar products (configurable via SEARCH_RESULT_LIMIT env var).
    Returns empty list if no products meet the similarity threshold.
    """
    with SEARCH_LATENCY.time(), tracer.start_as_current_span("search_products") as span:
        span.set_attribute("query", query.query)
        request_id = request.headers.get("X-Request-ID", str(time.time_ns()))
        log = logger.bind(request_id=request_id)

        try:
            log.info("search_request_received", query=query.query)

            # Generate embedding for search query
            query_embedding = get_embedding(query.query)
            log.debug("embedding_generated", embedding_size=len(query_embedding))

            # Search for similar products
            results = search_similar_products(query_embedding)
            log.info(
                "search_completed",
                results_count=len(results),
                categories=[r.get("category") for r in results],
            )

            SEARCH_REQUESTS.labels(status="success").inc()
            span.set_status(Status(StatusCode.OK))
            return results

        except HTTPException as he:
            SEARCH_REQUESTS.labels(status="error").inc()
            span.set_status(Status(StatusCode.ERROR), str(he))
            span.record_exception(he)
            log.error("search_failed", error=str(he), status_code=he.status_code)
            raise
        except Exception as e:
            SEARCH_REQUESTS.labels(status="error").inc()
            span.set_status(Status(StatusCode.ERROR), str(e))
            span.record_exception(e)
            log.error("search_failed_unexpectedly", error=str(e))
            raise HTTPException(status_code=500, detail="Search request failed")


# Initialize instrumentations
FastAPIInstrumentor.instrument_app(app)
LoggingInstrumentor().instrument()
Psycopg2Instrumentor().instrument()

# Initialize Prometheus instrumentator
Instrumentator().instrument(app).expose(app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
