# Dataset Processing Guide

This document describes the complete pipeline for processing benchmark datasets in LeanRAG, from raw data download to knowledge graph construction and querying.

## Overview

The LeanRAG pipeline transforms raw benchmark datasets into hierarchical knowledge graphs through a multi-stage process involving context extraction, chunking, entity/relation extraction, deduplication, and graph construction.

## Dataset Processing

### 1. Dataset Download and Storage

**Source**: [UltraDomain Dataset](https://huggingface.co/datasets/TommyChien/UltraDomain/tree/main)

**Storage Location**: `datasets/ultradomain/`

**File Naming**: `{dataset_prefix}.jsonl`
- Examples: `cs.jsonl`, `legal.jsonl`, `agriculture.jsonl`

**File Format**: Each line contains a JSON object with fields:
- `input`: Query or question
- `answers`: Ground truth answers  
- `context`: Textual content for knowledge extraction
- Other metadata fields

### 2. Context Extraction

**Script**: `extract_context.py`

**Input**: `datasets/ultradomain/` (all `.jsonl` files)

**Output**: `datasets/unique_contexts/{dataset_prefix}_unique_contexts.json`

**Process**:
- Processes every file in the ultradomain directory
- Extracts the `context` field from each JSON line
- Ensures context uniqueness across the dataset
- Removes duplicate content across different domains
- Stores processed contexts in standardized JSON format

**Command**:
```bash
python extract_context.py [--input datasets/ultradomain] [--output datasets/unique_contexts]
```

### 3. Data Chunking

**Script**: `file_chunk.py`

**Input**: `datasets/unique_contexts/` (all `*_unique_contexts.json` files)

**Output**: `data/{dataset_prefix}/chunk.json`

**Process**:
- Processes every file in the unique_contexts directory
- Splits contexts into optimally-sized chunks:
  - **Max token size**: 512 tokens
  - **Overlap**: 64 tokens (sliding window)
- Calculates hash codes for traceability
- Stores results as JSON with `hash_code` and `text` fields

**Output Structure**:
```json
[
  {
    "hash_code": "md5_hash_of_content",
    "text": "chunk content..."
  }
]
```

**Command**:
```bash
python file_chunk.py [--input datasets/unique_contexts] [--output data]
```

## Data Processing

### 1. Entity and Relation Extraction

**Script**: `process_chunk.py`

**Input**: `data/{dataset_prefix}/chunk.json`

**Output**: 
- `data/{dataset_prefix}/entity-extract.jsonl`
- `data/{dataset_prefix}/relation-extract.jsonl`

**Process**:
- Uses LLM (configured in `config.yaml`) to extract entities and relations from chunks
- Applies GraphRAG extraction methodology
- Extracts both entity descriptions and relationship information
- Stores raw extraction results in JSONL format

**Command**:
```bash
python process_chunk.py {dataset_prefix} -c config.yaml
```

### 2. Deduplication and Summarization

**Script**: `deal_triple.py`

**Input**: 
- `data/{dataset_prefix}/entity-extract.jsonl`
- `data/{dataset_prefix}/relation-extract.jsonl`

**Output**:
- `data/{dataset_prefix}/entity.jsonl`
- `data/{dataset_prefix}/relation.jsonl`

**Process**:
- Processes and deduplicates entities and relations from extraction step
- Uses LLM to summarize entity descriptions exceeding token thresholds
- Consolidates duplicate entities with similar names or descriptions
- Produces clean, deduplicated knowledge base components

**Command**:
```bash
python deal_triple.py --input data/{dataset_prefix} --output data/{dataset_prefix} -c config.yaml
```

### 3. Knowledge Graph Construction

**Script**: `build_graph.py`

**Input**:
- `data/{dataset_prefix}/entity.jsonl`
- `data/{dataset_prefix}/relation.jsonl`
- `data/{dataset_prefix}/chunk.json`

**Output**:
- `data/{dataset_prefix}/all_entities.json`
- `data/{dataset_prefix}/generate_relations.json` 
- `data/{dataset_prefix}/community.json`

**Process**:
1. **Standard Graph Creation**: Builds initial graph with vector embeddings
2. **Hierarchical Clustering**: Creates additional semantic aggregation levels using:
   - UMAP dimensionality reduction
   - Gaussian Mixture Models for clustering
   - Community detection algorithms
3. **Database Storage**:
   - **Qdrant**: Stores entity vectors for similarity search
   - **MySQL**: Creates database `{dataset_prefix}` with tables:
     - `entities`: Entity information and embeddings metadata
     - `relations`: Relationship data between entities  
     - `community`: Hierarchical community/cluster information
     - `text_units`: Original chunk data for evidence retrieval

**Command**:
```bash
python build_graph.py -p data/{dataset_prefix} -c config.yaml
```

## Querying

### Graph Querying

**Script**: `query_graph.py`

**Input**: Constructed knowledge graph (database + vector index)

**Process**:
- Performs hierarchical "bottom-up" retrieval
- Uses vector similarity search to find relevant entities
- Traverses graph structure from specific entities to general communities
- Aggregates evidence while minimizing redundancy
- Generates contextual responses using retrieved information

**Command**:
```bash
python query_graph.py -q "Your query here" -w data/{dataset_prefix} -c config.yaml --topk 10
```

## File Structure

After complete processing, the file structure looks like:

```
datasets/
├── ultradomain/           # Raw downloaded datasets
│   ├── cs.jsonl
│   ├── legal.jsonl
│   └── ...
├── unique_contexts/       # Extracted unique contexts  
│   ├── cs_unique_contexts.json
│   ├── legal_unique_contexts.json
│   └── ...
└── chunks/               # Legacy location (moved to data/)

data/
├── cs/                   # Computer Science dataset
│   ├── chunk.json
│   ├── entity-extract.jsonl
│   ├── relation-extract.jsonl
│   ├── entity.jsonl
│   ├── relation.jsonl
│   ├── all_entities.json
│   ├── generate_relations.json
│   └── community.json
├── legal/               # Legal dataset  
│   └── [same structure]
└── mix/                 # Mixed domain dataset
    └── [same structure]
```

## Database Schema

### MySQL Database: `{dataset_prefix}`

**Tables**:
- **`entities`**: Entity data with descriptions and metadata
- **`relations`**: Relationships between entities
- **`community`**: Hierarchical clustering results
- **`text_units`**: Original chunk data for evidence retrieval

### Qdrant Collections

**Collection**: `{dataset_prefix}_entities`
- Stores entity vector embeddings
- Enables fast similarity search
- Supports hierarchical retrieval queries

## Configuration

All processing steps use `config.yaml` for:
- LLM API settings (OpenAI, Groq, Lambda Labs, Cerebras)
- Database configuration (MySQL + Qdrant)  
- Processing parameters (token limits, concurrency, etc.)

## Performance Notes

- **Chunking**: 512 tokens optimal for most LLMs while maintaining context
- **Extraction**: Concurrent processing for faster entity/relation extraction
- **Graph Construction**: Hierarchical clustering creates 3-4 levels typically
- **Query Speed**: Sub-second response times for most knowledge graphs

## Troubleshooting

1. **Download Issues**: Check internet connection and HuggingFace availability
2. **Chunking Errors**: Verify unique_contexts files contain valid text
3. **Extraction Failures**: Test LLM configuration with `python tests/quick_test.py`
4. **Database Issues**: Ensure Docker services running with `./scripts/start_db.sh`
5. **Query Problems**: Verify graph construction completed successfully

## See Also

- [README.md](README.md) - General setup and usage
- [CLAUDE.md](CLAUDE.md) - Detailed development guide
- [config/](config/) - Configuration examples for different API providers