# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LeanRAG is a knowledge-graph-based Retrieval-Augmented Generation (RAG) framework that uses semantic aggregation and hierarchical retrieval. The system constructs multi-layer knowledge graphs from documents and performs efficient retrieval by traversing from fine-grained entities up through semantic aggregation layers.

## Common Development Commands

### Essential Testing Commands
```bash
# ALWAYS test configuration before running extraction/processing
python tests/quick_test.py

```

### Cost Estimation (NEW)
Estimate processing costs before running full extraction:
```bash
# Quick estimation
python tests/cost_estimator.py --dataset cs

```

**Key Features:**
- **Token estimation**: Analyzes chunk files to estimate token usage
- **Sample processing**: Runs extraction on small sample to measure actual costs
- **Full extrapolation**: Projects costs and processing time for complete dataset
- **Provider pricing**: Current rates for OpenAI, Groq, Cerebras, Lambda Labs
- **Rate limit analysis**: Shows impact of TPM limits on processing time
- **Processing time**: Estimates hours needed for full extraction

**Example Output:**
```
📁 Dataset: cs (2,289 chunks)
🤖 Model: groq/openai/gpt-oss-120b ($0.50/$0.80 per 1M tokens)
🧪 Sample: 3 chunks processed in 7.19s (2,113 avg tokens/chunk)
📈 Full Dataset: 4.8M tokens, ~1.5 hours, $3.87 estimated cost
⏱️  Rate Limits: 200K TPM limit adds minimal delay
```

### Test Infrastructure
The `tests/` directory provides comprehensive validation:
- **Configuration validation**: `quick_test.py`

### Package Management
This project uses pip with `requirements.txt`. No setuptools/pip-tools configuration is present.

## Development Setup

### Environment Setup
```bash
# Create virtual environment using uv (recommended)
pip install --upgrade pip
pip install uv
uv venv leanrag --python=3.10
source leanrag/bin/activate  # Unix/macOS
# leanrag\Scripts\activate   # Windows

# Alternative: conda
conda create -n leanrag python=3.10
conda activate leanrag

# Install dependencies
uv pip install -e .
```

### Configuration
- Edit `config.yaml` to configure model endpoints and API keys
- Main configuration sections:
  - `llm_conf`: LLM API settings (Lambda Labs, OpenAI, Groq, Cerebras)
  - `embedding_conf`: Embedding API settings (OpenAI recommended)
  - `database`: Database backend selection (supabase or mysql)
  - `model_params`: Token limits and embedding dimensions

### API Provider Support and Configuration

**NEW: Consolidated Configuration**
All components now use a single `config.yaml` file in the root directory.

**Quick Setup Commands:**
```bash
# For Lambda Labs LLM + OpenAI embedding (most common)
cp config/config_lambda_example.yaml config.yaml

# For full OpenAI setup (both LLM and embedding)
cp config/config_openai_full_example.yaml config.yaml

# For Groq LLM + OpenAI embedding (fast inference)
cp config/config_groq_example.yaml config.yaml

# For Cerebras LLM + OpenAI embedding (specialized hardware)
cp config/config_cerebras_example.yaml config.yaml

# Edit config.yaml with your API keys
# Test configuration
python tests/quick_test.py
```

## Database Configuration

LeanRAG supports two database backends:

### MySQL + Qdrant (Default)

**Database Configuration:**
```yaml
# Database Configuration
database:
  # Backend options: 
  # - "mysql" (Qdrant + MySQL) <- NEW: Recommended
  # - "supabase" (PostgreSQL + pgvector) 
  backend: "mysql"  # Using Supabase for relational data, Qdrant for vectors
  
  # Feature flags for database operations
  log_query_times: true  # Log query performance metrics
  enable_parallel_operations: true  # Enable parallel insertion for hybrid mode

# Qdrant Configuration (used when database.backend = "mysql")
qdrant_conf:
  host: "localhost"
  port: 6333
  timeout: 30
  # collection_prefix: "leanrag"  # Prefix for collection names (optional)

# MySQL Configuration (used when database.backend = "mysql")
mysql_conf:
  host: "localhost"
  port: 4321
  user: "root"
  password: "123"
  charset: "utf8mb4"
  # Database name will be auto-generated based on working directory name
  autocommit: true
  connect_timeout: 10
```
**Configuration Structure:**
The consolidated `config.yaml` contains all settings:
- `llm_conf`: LLM API settings (Lambda Labs, OpenAI, Groq, Cerebras)
- `embedding_conf`: Embedding API settings  
- `commonkg_conf`: CommonKG-specific parameters
- `graphextraction_conf`: GraphExtraction-specific parameters
- `model_params`: Token limits and embedding dimensions

**Rate Limiting (NEW):**
Automatic rate limiting prevents API quota exhaustion:
```yaml
llm_conf:
  enable_rate_limiting: true        # Enable/disable rate limiting
  rate_limit_tpm: 200000           # Tokens per minute limit
```

- **Groq**: 250K TPM free tier → set to 200K for safety
- **OpenAI**: ~1M TPM typical → set to 800K for safety  
- **Cerebras/Lambda**: Conservative 250-500K TPM limits
- Automatically waits when approaching limits
- Parses API error messages for optimal retry delays

**Setup Commands:**
```bash
# Start database services (Docker Compose)
./scripts/start_db.sh

# Build knowledge graph
python build_graph.py -p data/cs -c config.yaml

# Query
python query_graph.py -q "What is machine learning?" -w data/cs -c config.yaml

# Stop database services when done
./scripts/stop_db.sh
```

### Supabase (Future DB Option)
**Configuration:**
```yaml
# Database Configuration
database:
  backend: "supabase"
  
supabase_conf:
  url: "https://your-project.supabase.co"
  key: "your-supabase-anon-key-here"
  service_key: "your-supabase-service-role-key-here"  # For admin operations
  max_connections: 10
  timeout: 30
  retry_attempts: 3
```

**Environment Variables (recommended):**
```bash
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_ANON_KEY="your-anon-key"
export SUPABASE_SERVICE_KEY="your-service-key"  # For admin operations
```


## Core Pipeline Commands

### 1. Document Chunking
```bash
python file_chunk.py
```
Splits documents into 1024-token chunks with 128-token sliding windows.

### 2. Knowledge Graph Extraction

> **Choosing the Right Method**: Use `python extraction_benchmark.py` to get recommendations for your document type, or see `EXTRACTION_METHOD_GUIDE.md` for detailed guidance.

**GraphRAG (LLM-based extraction) - Primary Method**
```bash
# Extract entities and relations (creates data/{prefix}/ directory)
python process-chunk.py {dataset_prefix} -c config.yaml

# Process and deduplicate results (reads and writes to same directory)
python deal_triple.py --input data/{dataset_prefix} --output data/{dataset_prefix} -c config.yaml
```

**Note**: All scripts now use API-compatible versions only. Local LLM support has been removed.

### 3. Graph Construction

```bash
# Build hierarchical knowledge graph 
python build_graph.py -p data/output_dir -c config.yaml
```
Builds hierarchical knowledge graph with semantic clustering and aggregation.

### 4. Query and Retrieval

```bash
# Query the knowledge graph
python query_graph.py -q "Your query here" -w data/output_dir -c config.yaml --topk 10

# Example query
python query_graph.py -q "What is machine learning?" -w data/cs -c config.yaml --topk 5
```
Performs hierarchical retrieval starting from relevant entities and traversing up the knowledge graph.

## Architecture Overview

### Core Components

**Knowledge Graph Construction (`build_graph.py`)**
- Processes extracted entities and relations from Step 2
- Performs hierarchical clustering using `_cluster_utils.py`
- Creates multi-layer graph structure stored in MySQL/ChromaDB databases
- Entry point: `get_common_rag_res()` function
- Supports Lambda Labs and OpenAI APIs for both LLM and embeddings
- Supports OpenAI embedding models (text-embedding-3-large, text-embedding-3-small, text-embedding-ada-002)

**Retrieval System (`query_graph.py`)**
- Implements hierarchical "bottom-up" retrieval strategy
- Uses vector similarity search for entity matching
- Traverses graph structure to gather contextual evidence
- Supports Lambda Labs and OpenAI APIs for both LLM and embeddings
- Key functions: `embedding()`, vector search, and path generation

**Database Layer (`database_utils_mysql.py` + `database_utils_qdrant.py` + `database_utils_supabase.py`)**
- Dual backend support: MySQL+Qdrant (default) or unified Supabase (PostgreSQL+pgvector)
- Qdrant/pgvector: Vector database for entity embeddings
- MySQL/PostgreSQL: Graph structure and relationships storage
- Functions: `build_vector_search()`, `search_vector_search()`, `find_path()`

**Clustering Utilities (`cluster_utils.py`)**
- Implements `Hierarchical_Clustering` class for semantic aggregation
- Uses UMAP dimensionality reduction and Gaussian Mixture Models
- Creates tree-structured knowledge representation

### Data Flow
1. **Chunking**: Documents → text chunks (`file_chunk.py`)
2. **Extraction**: Chunks → entities/relations (CommonKG or GraphRAG)
3. **Clustering**: Entities → hierarchical clusters (`build_graph.py`)
4. **Storage**: Graph → Database (MySQL+Qdrant or Supabase)
5. **Retrieval**: Query → relevant paths → context (`query_graph.py`)

### Key Directories
- `data/`: Output directory for GraphRAG extraction results
- `tests/`: Test scripts for configuration validation and debugging
- `utils/`: Core utilities (clustering, database, LLM management, I/O)
- `scripts/`: Database migration and benchmark scripts
- `datasets/`: Benchmark datasets and evaluation results


### Configuration Notes
- All settings configured via single `config.yaml` file in root directory
- **Supported LLM providers**: Lambda Labs, OpenAI, Groq, Cerebras
- **Supported embedding providers**: OpenAI (recommended)
- Default embedding: OpenAI text-embedding-3-small (1536 dimensions)
- Database files created in working directory during graph construction

### Important Development Practices

**Always Test Configuration First**: Before running any extraction or processing, test the configuration:
```bash
python tests/quick_test.py  
```

**API Key Security**: 
- Never commit API keys to the repository
- Use the `config/*_example.yaml` files as templates
- Keep actual `config.yaml` files out of version control

**Working Directory Dependencies**: 
- Database files are created in the current working directory during graph construction
- Ensure you're in the correct directory when running commands
- Output files are typically created relative to the working directory

### Database Dependencies

**Option 1: MySQL + Qdrant (Default)**
- **Qdrant**: Vector database for entity embeddings (Docker-based setup)
- **MySQL**: Graph structure storage (Docker-based setup)
- **Docker Compose**: Automated setup via `./scripts/start_db.sh`
- **Management**: Use `./scripts/stop_db.sh` to stop services
- Database services run on localhost:6333 (Qdrant) and localhost:4321 (MySQL)

**Option 2: Supabase (Future?)**
- **Supabase**: PostgreSQL + pgvector for both vector and relational data
- Single database for all operations - no separate installations needed
- Cloud-hosted was tested but found to be too slow. May test with local Docker install in the future

### Database Management

**Docker Compose Commands:**
```bash
# Start databases (MySQL + Qdrant)
./scripts/start_db.sh

# Stop databases  
./scripts/stop_db.sh

# Check database status
docker compose ps

# View logs
docker compose logs mysql
docker compose logs qdrant

# Restart services if needed
docker compose restart mysql
docker compose restart qdrant
```

**Database URLs:**
- MySQL: `localhost:4321` (user: root, password: 123)
- Qdrant: `localhost:6333` (REST API endpoint)

### Performance and Benchmarking
- Benchmarks available in `datasets/` folder (cs, legal, agriculture, mixed domains)
- Results demonstrate ~46% reduction in retrieval redundancy vs flat retrieval

### Common Issues and Debugging

**Configuration Problems**: 
- If extraction fails, first run the relevant `quick_test.py` scripts
- Check API keys are correctly set in all config files
- Verify model names match the API provider's available models

**Pipeline Issues**:
- Ensure output from one step exists before running the next step
- Check file paths and directory permissions
- Database connection issues usually indicate MySQL configuration problems in `utils/database_utils.py`

**Performance Issues**:
- Large documents may require adjusting chunk sizes in `file_chunk.py`
- Concurrent request limits can be modified in config files (`max_concurrent` setting)
- Memory issues during clustering can be resolved by reducing batch sizes

### File Dependencies and Data Flow
```
file_chunk.py → chunks.json (stored in data/{prefix})
        ↓
process-chunk.py + deal_triple.py → data/{prefix}/entity.jsonl, relation.jsonl  
        ↓
build_graph.py → Database storage (ChromaDB+MySQL)
        ↓
query_graph.py → final answers
```

## Recent Changes

### Configuration Consolidation
- **Single config file**: All settings now use `config.yaml` in root directory 
- **Simplified setup**: Just copy one example file and edit API keys

### API-First Architecture  
- **Unified scripts**: Main pipeline scripts (`build_graph.py`, `query_graph.py`) now support multiple API providers
- **API-only**: System supports Lambda Labs, OpenAI, Groq, and Cerebras APIs
- **Removed**: All local LLM configurations (Ollama, vLLM) for simplicity

### New LLM Provider Support
- **Groq API**: Ultra-fast inference with specialized hardware
  - **Models**: llama3-70b-8192, mixtral-8x7b-32768, etc.
  - **Benefits**: Extremely fast response times, high throughput
  - **Setup**: Copy `config_groq_example.yaml` to `config.yaml`

- **Cerebras API**: Fast inference with specialized AI hardware
  - **Models**: llama3.1-70b, llama3.1-8b, etc.
  - **Benefits**: Fast inference, good for large models
  - **Setup**: Copy `config_cerebras_example.yaml` to `config.yaml`