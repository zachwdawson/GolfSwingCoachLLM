.PHONY: help dev test fmt lint compose-up compose-down seed

help:
	@echo "Available commands:"
	@echo "  make dev          - Start development environment"
	@echo "  make test         - Run all tests"
	@echo "  make fmt          - Format code"
	@echo "  make lint         - Lint code"
	@echo "  make compose-up   - Start Docker Compose services"
	@echo "  make compose-down - Stop Docker Compose services"
	@echo "  make seed         - Seed database with sample data"

dev:
	@echo "Starting development environment..."
	@make compose-up

test:
	@echo "Running backend tests..."
	cd backend && if [ -d venv ]; then source venv/bin/activate && pytest; else python3 -m pytest; fi
	@echo "Running frontend type check..."
	cd frontend && npm run type-check

fmt:
	@echo "Formatting backend code..."
	cd backend && ruff format .
	@echo "Formatting frontend code..."
	cd frontend && npm run format

lint:
	@echo "Linting backend code..."
	cd backend && ruff check . && mypy app
	@echo "Linting frontend code..."
	cd frontend && npm run lint

compose-up:
	docker compose up -d
	@echo "Services started. Backend: http://localhost:8000, Frontend: http://localhost:3000"

compose-down:
	docker compose down
	@echo "Services stopped"

seed:
	@echo "Seeding database..."
	@echo "TODO: Implement seed script"

