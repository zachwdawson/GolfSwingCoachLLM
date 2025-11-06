# File Tree

```
.
├── .github/
│   └── workflows/
│       └── ci-cd.yml              # GitHub Actions CI/CD workflow
├── backend/                       # FastAPI backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI app entry point
│   │   └── tests/
│   │       ├── __init__.py
│   │       └── test_main.py       # Basic tests
│   ├── .env.example               # Backend environment variables template
│   ├── Dockerfile                 # Backend Docker image
│   └── pyproject.toml             # Python dependencies and tool config (ruff, mypy, pytest)
├── frontend/                      # Next.js 14 frontend
│   ├── app/
│   │   ├── globals.css            # Global styles
│   │   ├── layout.tsx             # Root layout
│   │   ├── page.tsx               # Home page
│   │   └── upload/
│   │       └── page.tsx           # File upload page
│   ├── .env.example               # Frontend environment variables template
│   ├── .eslintrc.json             # ESLint configuration
│   ├── .prettierrc                # Prettier configuration
│   ├── Dockerfile                 # Frontend Docker image
│   ├── next.config.js             # Next.js configuration
│   ├── package.json               # Node.js dependencies
│   └── tsconfig.json              # TypeScript configuration
├── infra/                         # Terraform infrastructure
│   ├── main.tf                    # Main Terraform resources (S3, RDS, ECR, ECS, ALB)
│   ├── variables.tf               # Terraform variables
│   └── README.md                  # Infrastructure documentation
├── data/                          # Data files
│   └── drills/
│       └── example.yaml           # Sample drill card YAML
├── .gitignore                     # Git ignore patterns
├── docker-compose.yml             # Local development Docker Compose setup
├── Makefile                       # Development commands (dev, test, fmt, lint, compose-up/down)
└── README.md                      # Project documentation and local run instructions
```

## Key Files

- **docker-compose.yml**: Defines Postgres, backend, and frontend services for local development
- **Makefile**: Provides convenient commands for development workflow
- **backend/pyproject.toml**: Python project configuration with ruff, mypy, and pytest
- **frontend/package.json**: Node.js dependencies with Next.js 14, TypeScript, ESLint, Prettier
- **infra/main.tf**: Terraform configuration for AWS resources (S3, RDS, ECR, ECS, ALB)
- **.github/workflows/ci-cd.yml**: CI/CD pipeline for testing and deployment

