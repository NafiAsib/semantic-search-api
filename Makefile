.PHONY: help db-create db-drop db-reset db-connect db-status db-create-index docker-up docker-down docker-restart clean

# Load environment variables from .env file
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

# Default target
help:
	@echo "Available commands:"
	@echo "  make db-create       - Create database schema and extensions"
	@echo "  make db-create-index - Create vector index (run AFTER inserting data)"
	@echo "  make db-drop         - Drop all tables (WARNING: destroys data)"
	@echo "  make db-reset        - Drop and recreate database (WARNING: destroys data)"
	@echo "  make db-connect      - Connect to database via psql"
	@echo "  make db-status       - Show database status and table info"
	@echo "  make docker-up       - Start all Docker services"
	@echo "  make docker-down     - Stop all Docker services"
	@echo "  make docker-restart  - Restart all Docker services"
	@echo "  make clean           - Clean up Docker volumes (WARNING: destroys data)"

# Start Docker services
docker-up:
	@echo "Starting Docker services..."
	docker compose up -d
	@echo "Waiting for PostgreSQL to be ready..."
	@sleep 5
	@echo "Services started successfully!"

# Stop Docker services
docker-down:
	@echo "Stopping Docker services..."
	docker compose down

# Restart Docker services
docker-restart:
	@echo "Restarting Docker services..."
	docker compose restart

# Create database schema
db-create:
	@echo "Creating database schema..."
	@docker exec -i semantrix psql -U $(DB_USER) -d $(DB_NAME) < db.sql
	@echo "Database schema created successfully!"

# Create vector index (run after inserting data for better performance)
db-create-index:
	@echo "Creating vector index (this may take a while)..."
	@docker exec -i semantrix psql -U $(DB_USER) -d $(DB_NAME) -c "CREATE INDEX IF NOT EXISTS idx_products_embedding_ivfflat ON products USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"
	@echo "Vector index created successfully!"

# Drop all tables
db-drop:
	@echo "WARNING: This will drop all tables and destroy all data!"
	@read -p "Are you sure? (yes/no): " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		echo "Dropping tables..."; \
		docker exec -i semantrix psql -U $(DB_USER) -d $(DB_NAME) -c "DROP TABLE IF EXISTS products CASCADE;"; \
		echo "Tables dropped successfully!"; \
	else \
		echo "Operation cancelled."; \
	fi

# Reset database (drop and create)
db-reset: db-drop db-create
	@echo "Database reset complete!"

# Connect to database
db-connect:
	@docker exec -it semantrix psql -U $(DB_USER) -d $(DB_NAME)

# Show database status
db-status:
	@echo "Database Status:"
	@echo "================"
	@docker exec -i semantrix psql -U $(DB_USER) -d $(DB_NAME) -c "\dt"
	@echo ""
	@echo "Products Table Count:"
	@docker exec -i semantrix psql -U $(DB_USER) -d $(DB_NAME) -c "SELECT COUNT(*) FROM products;" 2>/dev/null || echo "Table does not exist yet"
	@echo ""
	@echo "Extensions:"
	@docker exec -i semantrix psql -U $(DB_USER) -d $(DB_NAME) -c "\dx"

# Clean up Docker volumes
clean:
	@echo "WARNING: This will remove all Docker volumes and destroy all data!"
	@read -p "Are you sure? (yes/no): " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		echo "Stopping services..."; \
		docker compose down -v; \
		echo "Removing local data directories..."; \
		rm -rf postgres_data; \
		echo "Cleanup complete!"; \
	else \
		echo "Operation cancelled."; \
	fi
