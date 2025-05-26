#!/bin/bash

# Create directory structure
mkdir -p prometheus
mkdir -p grafana/provisioning/datasources
mkdir -p grafana/provisioning/dashboards
mkdir -p grafana/dashboards

# Install Python dependencies for observability
pip install opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-instrumentation-fastapi \
    opentelemetry-instrumentation-logging \
    opentelemetry-instrumentation-psycopg2 \
    opentelemetry-exporter-otlp \
    prometheus-client \
    prometheus-fastapi-instrumentator \
    structlog

echo "Directory structure created and dependencies installed!"
echo "Please ensure your configuration files are in place:"
echo "- prometheus/prometheus.yml"
echo "- grafana/provisioning/datasources/datasource.yml"
echo "- grafana/provisioning/dashboards/dashboards.yml"
echo "- grafana/dashboards/product_search.json" 