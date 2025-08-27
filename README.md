# LeanRAG: Knowledge-Graph-Based Generation with Semantic Aggregation and Hierarchical Retrieval

[![Python Version](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  [![arXiv](https://img.shields.io/badge/arXiv-2508.10391-b31b1b.svg)](https://arxiv.org/abs/2508.10391)[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

LeanRAG is an efficient, open-source framework for Retrieval-Augmented Generation, leveraging knowledge graph structures with semantic aggregation and hierarchical retrieval to generate context-aware, concise, and high-fidelity responses.

## ✨ Features

- **Semantic Aggregation**: Clusters entities into semantically coherent summaries and constructs explicit relations to form a navigable aggregation-level knowledge network.
- **Hierarchical, Structure-Guided Retrieval**: Initiates retrieval from fine-grained entities and traverses up the knowledge graph to gather rich, highly relevant evidence efficiently.
- **Reduced Redundancy**: Optimizes retrieval paths to significantly reduce redundant information—LeanRAG achieves ~46% lower retrieval redundancy compared to flat retrieval baselines (based on benchmark evaluations).
- **Benchmark Performance**: Demonstrates superior performance across multiple QA benchmarks with improved response quality and retrieval efficiency.

## 🏛️ Architecture Overview
![Overview of LeanRAG](pic/framework.png)

LeanRAG’s processing pipeline follows these core stages:

1. **Semantic Aggregation**  
   - Group low-level entities into clusters; generate summary nodes and build adjacency relations among them for efficient navigation.

2. **Knowledge Graph Construction**  
   - Construct a multi-layer graph where nodes represent entities and aggregated summaries, with explicit inter-node relations for graph-based traversal.

3. **Query Processing & Hierarchical Retrieval**  
   - Anchor queries at the most relevant detailed entities ("bottom-up"), then traverse upward through the semantic aggregation graph to collect evidence spans.

4. **Redundancy-Aware Synthesis**  
   - Streamline retrieval paths and avoid overlapping content, ensuring concise evidence aggregation before generating responses.

5. **Generation**  
   - Use retrieved, well-structured evidence as input to an LLM to produce coherent, accurate, and contextually grounded answers.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+  
- Docker & Docker Compose (for local databases)
- MySQL 8.0+ or Supabase account (database backend)

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/jeffjacobsen/LeanRAG.git
    cd LeanRAG
    ```

2. **Create a virtual environment:**
    ```bash
    # Upgrade pip and install uv (recommended)
    pip install --upgrade pip
    pip install uv

    # Create and activate a virtual environment using uv
    uv venv leanrag --python=3.10
    source leanrag/bin/activate      # For Unix/macOS
    leanrag\Scripts\activate         # For Windows

    # Alternatively, you can use conda to create and activate the environment
    conda create -n leanrag python=3.10
    conda activate leanrag
    ```

3. **Install the required dependencies:**
    ```bash
    uv pip install -e .
    ```

4. **Start the database services:**
    ```bash
    # Start both Qdrant (vector DB) and MySQL (relational DB) using Docker
    ./scripts/start_db.sh

    # Check service health
    docker compose ps
    ```

## 🧪 Configuration & Testing

LeanRAG supports multiple API providers. Choose your setup:


### **API Provider Options**
```bash
# Lambda Labs (cost-effective)
cp config/config_lambda_example.yaml config.yaml

# OpenAI (high quality)
cp config/config_openai_full_example.yaml config.yaml

# Groq (ultra-fast inference)  
cp config/config_groq_example.yaml config.yaml

# Cerebras (specialized hardware)
cp config/config_cerebras_example.yaml config.yaml
```

### **Test Your Configuration**
```bash

# Test API and database connectivity
python tests/quick_test.py

# Estimate processing costs
python tests/cost_estimator.py --dataset cs --sample-size 5
```

## 💻 Usage Workflow

### **Quick Start Pipeline**
```bash
# Process your dataset (replace 'cs' with your dataset name)
python process_chunk.py cs       # Extract entities and relations
python deal_triple.py cs         # Deduplicate and clean
python build_graph.py cs         # Build knowledge graph
python query_graph.py cs -q "Your question here"  # Query the graph
```

Here's the detailed pipeline flow:

### **Step 1: Document Chunking**
In `file_chunk.py`, split the document into chunks:

- **Chunk size**: `1024`
- **Sliding step**: `128` (i.e., use a sliding window with step 128)

Each dictionary in the resulting `chunk` file contains two attributes:

- `hash_code`: hash calculated from the `text` content for traceability
- `text`: the chunk text content

---

### **Step 2: Extract Triples and Entity Descriptions**

#### **GraphRAG Extraction**
LeanRAG uses GraphRAG for knowledge extraction, which relies on LLM capability to perform few-shot extraction with given examples in the prompt.

**Usage:**

**API-compatible version (recommended):**
1. Configure API provider (already done in Quick Setup above)

2. Test configuration:
   ```bash
   python tests/quick_test.py
   ```

3. Run extraction:
   ```bash
   python process_chunk.py {dataset_prefix}
   python deal_triple.py {dataset_prefix}
   ```

**Outputs include:**
- entity.jsonl
- relation.jsonl

**Features:**
- Multiple API provider support (Lambda Labs, OpenAI, Groq, Cerebras)
- Async processing with configurable concurrency
- Rate limiting and retry logic
- Cost tracking and progress monitoring

### **Step 3: Build the Knowledge Graph**

```bash
# Build hierarchical knowledge graph with semantic clustering
python build_graph.py {dataset_prefix}
```

This step:
- Clusters extracted entity and relation descriptions
- Generates semantic aggregation layers  
- Constructs tree-structured knowledge graph
- Stores graph in selected database backend (MySQL or Supabase)
- Builds vector index using Qdrant for similarity search

### **Step 4: Query and Retrieval**

```bash
# Query the knowledge graph with hierarchical retrieval
python query_graph.py {dataset_prefix} -q "Your query here" --topk 10

# Example query
python query_graph.py cs -q "What is machine learning?" --topk 5
```

This performs:
- Vector similarity search to find relevant entities
- "Bottom-up" traversal through knowledge graph layers
- Evidence aggregation with redundancy reduction
- LLM-powered response generation

## 🏗️ Architecture & Performance

### **Database Backends**

**MySQL + Qdrant (Default)**
- **MySQL**: Relational data (entities, relations, communities, text chunks)
- **Qdrant**: Vector similarity search for entity embeddings
- **Performance**: 141× faster database operations vs cloud solutions
- **Setup**: Fully automated with Docker Compose

**Supabase Alternative**
- **PostgreSQL + pgvector**: Unified database for all operations
- **Benefits**: Cloud-hosted, automatic backups, managed infrastructure
- **Use case**: Production deployments requiring managed services

### **Performance Benchmarks**
- **46% reduction** in retrieval redundancy vs flat baselines
- **141× faster** database queries with local MySQL setup
- **5× faster** end-to-end processing vs cloud-only solutions
- Sub-second query response times for most knowledge graphs

### **Supported API Providers**
- **Lambda Labs**: Cost-effective, good performance
- **OpenAI**: Highest quality, comprehensive model support
- **Groq**: Ultra-fast inference with specialized hardware
- **Cerebras**: Fast inference, good for large models

## 🛠️ Database Management

```bash
# Start databases
./scripts/start_db.sh

# Stop databases  
./scripts/stop_db.sh

# Check database status
docker compose ps

# View logs
docker compose logs mysql
docker compose logs qdrant
```

## 📊 Testing & Debugging

```bash
# Test configuration and connectivity
python tests/quick_test.py

# Cost estimation before processing
python tests/cost_estimator.py --dataset cs --sample-size 5
```

## 📚 Citation

If you find LeanRAG useful in your research, please cite:

```bibtex
@article{leanrag2024,
  title={LeanRAG: Knowledge-Graph-Based Generation with Semantic Aggregation and Hierarchical Retrieval},
  author={Your Authors},
  journal={arXiv preprint arXiv:2508.10391},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Forked from:** https://github.com/RaZzzyz/LeanRAG.git
