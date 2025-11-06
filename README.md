# Golf Swing Coach MVP

A monorepo for an AI-powered golf swing analysis and coaching application.

## Project Structure

```
.
├── backend/          # FastAPI backend (Python 3.11)
├── frontend/         # Next.js 14 frontend (TypeScript)
├── infra/            # Terraform infrastructure
├── data/             # YAML drill cards
├── docker-compose.yml
├── Makefile
└── README.md
```

See [FILE_TREE.md](FILE_TREE.md) for a detailed file tree.

## Local Development

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local backend development)
- Node.js 20+ (for local frontend development)

### Quick Start

1. **Start all services with Docker Compose:**
   ```bash
   make compose-up
   ```
   This starts:
   - Postgres database on port 5432
   - Backend API on http://localhost:8000
   - Frontend app on http://localhost:3000

2. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Health: http://localhost:8000/health

3. **Stop services:**
   ```bash
   make compose-down
   ```

### Development Commands

- `make dev` - Start development environment (same as compose-up)
- `make test` - Run all tests (backend pytest, frontend type-check)
- `make fmt` - Format code (ruff for backend, prettier for frontend)
- `make lint` - Lint code (ruff+mypy for backend, ESLint for frontend)
- `make compose-up` - Start Docker Compose services
- `make compose-down` - Stop Docker Compose services

### Backend Development

1. **Set up environment:**
   ```bash
   cd backend
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Install dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Run locally:**
   ```bash
   uvicorn app.main:app --reload
   ```

### Frontend Development

1. **Set up environment:**
   ```bash
   cd frontend
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Run locally:**
   ```bash
   npm run dev
   ```

## Environment Variables

### Backend (.env)

- `DB_URL` - PostgreSQL connection string
- `AWS_REGION` - AWS region for S3
- `S3_BUCKET` - S3 bucket name for videos
- `USE_SSM` - Use AWS SSM Parameter Store (true/false)
- `EMBEDDINGS_API_KEY` - API key for embeddings service
- `LLM_API_KEY` - API key for LLM service
- `LOG_LEVEL` - Logging level (INFO, DEBUG, etc.)

### Frontend (.env)

- `NEXT_PUBLIC_API_BASE` - Backend API base URL (default: http://localhost:8000)

## Testing

- **Backend:** `pytest` (unit and integration tests)
- **Frontend:** `npm run type-check` (TypeScript type checking)
- **E2E:** Cypress tests (to be added)

## Infrastructure

Terraform configuration in `infra/` for AWS resources:
- S3 bucket for video storage
- RDS Postgres with pgvector
- ECR repositories
- ECS Fargate cluster
- Application Load Balancer

See `infra/README.md` for deployment instructions.

## CI/CD

GitHub Actions workflow (`.github/workflows/ci-cd.yml`) runs:
- Backend: ruff, mypy, pytest
- Frontend: ESLint, type-check, build
- Deploy: Build and push Docker images to ECR, update ECS services

## License

MIT
