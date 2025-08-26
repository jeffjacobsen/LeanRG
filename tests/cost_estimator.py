#!/usr/bin/env python3
"""
Cost estimation utility for LeanRAG API usage.
Estimates costs for different API providers based on token usage.
"""

import argparse
import yaml
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tiktoken

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# API pricing per 1M tokens (as of 2024)
API_PRICING = {
    "openai": {
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
        "text-embedding-3-large": {"input": 0.13, "output": 0.0},
        "text-embedding-3-small": {"input": 0.02, "output": 0.0},
        "text-embedding-ada-002": {"input": 0.10, "output": 0.0}
    },
    "groq": {
        "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
        "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
        "mixtral-8x7b-32768": {"input": 0.27, "output": 0.27},
        "gemma-7b-it": {"input": 0.07, "output": 0.07}
    },
    "lambda": {
        "hermes-3-llama-3.1-405b": {"input": 2.7, "output": 2.7},
        "llama-3.1-70b-instruct": {"input": 0.35, "output": 0.4},
        "llama-3.1-8b-instruct": {"input": 0.075, "output": 0.075}
    },
    "cerebras": {
        "llama3.1-70b": {"input": 0.60, "output": 0.60},
        "llama3.1-8b": {"input": 0.10, "output": 0.10}
    }
}

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        raise

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text using tiktoken."""
    try:
        # Map model names to tiktoken encodings
        model_encodings = {
            "gpt-4": "cl100k_base",
            "gpt-4-turbo": "cl100k_base", 
            "gpt-4o": "cl100k_base",
            "gpt-4o-mini": "cl100k_base",
            "gpt-3.5-turbo": "cl100k_base",
            "text-embedding-3-large": "cl100k_base",
            "text-embedding-3-small": "cl100k_base",
            "text-embedding-ada-002": "cl100k_base"
        }
        
        encoding_name = model_encodings.get(model, "cl100k_base")
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception:
        # Fallback to rough estimation: ~4 chars per token
        return len(text) // 4

def estimate_extraction_cost(data_dir: str, config: dict) -> Dict[str, float]:
    """Estimate cost for knowledge graph extraction phase."""
    
    # Load chunks to analyze
    chunks_file = Path(data_dir) / "chunk.json"
    if not chunks_file.exists():
        print(f"Warning: {chunks_file} not found, using estimated values")
        num_chunks = 1000
        avg_chunk_size = 1024
    else:
        try:
            with open(chunks_file, 'r') as f:
                chunks = json.load(f)
            num_chunks = len(chunks)
            avg_chunk_size = sum(count_tokens(chunk.get('text', '')) for chunk in chunks) // num_chunks if chunks else 1024
        except Exception:
            print(f"Error reading {chunks_file}, using estimated values")
            num_chunks = 1000
            avg_chunk_size = 1024
    
    llm_config = config.get('llm_conf', {})
    provider = llm_config.get('api_provider', 'openai').lower()
    model = llm_config.get('llm_model', 'gpt-4o-mini')
    
    # Estimate tokens for extraction prompts
    extraction_prompt_tokens = 500  # System prompt + instruction tokens
    input_tokens_per_chunk = extraction_prompt_tokens + avg_chunk_size
    output_tokens_per_chunk = 200  # Estimated extracted entities/relations
    
    total_input_tokens = input_tokens_per_chunk * num_chunks
    total_output_tokens = output_tokens_per_chunk * num_chunks
    
    # Calculate cost
    if provider in API_PRICING and model in API_PRICING[provider]:
        pricing = API_PRICING[provider][model]
        input_cost = (total_input_tokens / 1_000_000) * pricing["input"]
        output_cost = (total_output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
    else:
        print(f"Warning: Pricing not available for {provider}/{model}")
        total_cost = 0.0
        input_cost = 0.0
        output_cost = 0.0
    
    return {
        "provider": provider,
        "model": model,
        "num_chunks": num_chunks,
        "avg_chunk_tokens": avg_chunk_size,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }

def estimate_embedding_cost(data_dir: str, config: dict) -> Dict[str, float]:
    """Estimate cost for embedding generation."""
    
    # Estimate number of entities/relations to embed
    entities_file = Path(data_dir) / "entitiy.json"
    if entities_file.exists():
        try:
            with open(entities_file, 'r') as f:
                entities = json.load(f)
            num_entities = len(entities)
        except Exception:
            num_entities = 5000  # Default estimate
    else:
        num_entities = 5000  # Default estimate
    
    embedding_config = config.get('embedding_conf', {})
    provider = embedding_config.get('api_provider', 'openai').lower()
    model = embedding_config.get('model', 'text-embedding-3-small')
    
    # Estimate embedding input tokens
    avg_entity_length = 50  # Average tokens per entity/relation
    total_embedding_tokens = num_entities * avg_entity_length
    
    # Calculate cost
    if provider in API_PRICING and model in API_PRICING[provider]:
        pricing = API_PRICING[provider][model]
        embedding_cost = (total_embedding_tokens / 1_000_000) * pricing["input"]
    else:
        print(f"Warning: Embedding pricing not available for {provider}/{model}")
        embedding_cost = 0.0
    
    return {
        "provider": provider,
        "model": model,
        "num_entities": num_entities,
        "total_embedding_tokens": total_embedding_tokens,
        "embedding_cost": embedding_cost
    }

def estimate_query_cost(config: dict, num_queries: int = 100) -> Dict[str, float]:
    """Estimate cost for query processing."""
    
    llm_config = config.get('llm_conf', {})
    provider = llm_config.get('api_provider', 'openai').lower()
    model = llm_config.get('llm_model', 'gpt-4o-mini')
    
    # Estimate tokens per query
    query_tokens = 50  # Average query length
    context_tokens = 2000  # Retrieved context
    system_prompt_tokens = 300  # System instructions
    
    input_tokens_per_query = query_tokens + context_tokens + system_prompt_tokens
    output_tokens_per_query = 300  # Generated response
    
    total_input_tokens = input_tokens_per_query * num_queries
    total_output_tokens = output_tokens_per_query * num_queries
    
    # Calculate cost
    if provider in API_PRICING and model in API_PRICING[provider]:
        pricing = API_PRICING[provider][model]
        input_cost = (total_input_tokens / 1_000_000) * pricing["input"]
        output_cost = (total_output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
    else:
        print(f"Warning: Pricing not available for {provider}/{model}")
        total_cost = 0.0
        input_cost = 0.0
        output_cost = 0.0
    
    return {
        "provider": provider,
        "model": model,
        "num_queries": num_queries,
        "input_tokens_per_query": input_tokens_per_query,
        "output_tokens_per_query": output_tokens_per_query,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }

def print_cost_breakdown(extraction_cost: Dict, embedding_cost: Dict, query_cost: Dict):
    """Print detailed cost breakdown."""
    
    print("=" * 80)
    print("LEANRAG COST ESTIMATION BREAKDOWN")
    print("=" * 80)
    
    print("\n📊 EXTRACTION PHASE")
    print("-" * 40)
    print(f"Provider: {extraction_cost['provider']}")
    print(f"Model: {extraction_cost['model']}")
    print(f"Number of chunks: {extraction_cost['num_chunks']:,}")
    print(f"Average tokens per chunk: {extraction_cost['avg_chunk_tokens']}")
    print(f"Total input tokens: {extraction_cost['total_input_tokens']:,}")
    print(f"Total output tokens: {extraction_cost['total_output_tokens']:,}")
    print(f"Input cost: ${extraction_cost['input_cost']:.4f}")
    print(f"Output cost: ${extraction_cost['output_cost']:.4f}")
    print(f"Total extraction cost: ${extraction_cost['total_cost']:.4f}")
    
    print("\n🔮 EMBEDDING PHASE")
    print("-" * 40)
    print(f"Provider: {embedding_cost['provider']}")
    print(f"Model: {embedding_cost['model']}")
    print(f"Number of entities: {embedding_cost['num_entities']:,}")
    print(f"Total embedding tokens: {embedding_cost['total_embedding_tokens']:,}")
    print(f"Total embedding cost: ${embedding_cost['embedding_cost']:.4f}")
    
    print("\n❓ QUERY PHASE (per 100 queries)")
    print("-" * 40)
    print(f"Provider: {query_cost['provider']}")
    print(f"Model: {query_cost['model']}")
    print(f"Input tokens per query: {query_cost['input_tokens_per_query']}")
    print(f"Output tokens per query: {query_cost['output_tokens_per_query']}")
    print(f"Total input tokens (100 queries): {query_cost['total_input_tokens']:,}")
    print(f"Total output tokens (100 queries): {query_cost['total_output_tokens']:,}")
    print(f"Input cost: ${query_cost['input_cost']:.4f}")
    print(f"Output cost: ${query_cost['output_cost']:.4f}")
    print(f"Total query cost (100 queries): ${query_cost['total_cost']:.4f}")
    
    print("\n💰 TOTAL COST SUMMARY")
    print("=" * 40)
    setup_cost = extraction_cost['total_cost'] + embedding_cost['embedding_cost']
    per_query_cost = query_cost['total_cost'] / query_cost['num_queries']
    
    print(f"One-time setup cost: ${setup_cost:.4f}")
    print(f"Cost per query: ${per_query_cost:.6f}")
    print(f"Cost for 1,000 queries: ${per_query_cost * 1000:.4f}")
    print(f"Cost for 10,000 queries: ${per_query_cost * 10000:.4f}")

def main():
    parser = argparse.ArgumentParser(
        description="Estimate API costs for LeanRAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-c', '--config',
                       default='config.yaml',
                       help='Configuration file path (default: config.yaml)')
    parser.add_argument('-d', '--dataset',
                       default='cs',
                       help='Data directory path (default: data/output_dir)')
    parser.add_argument('-q', '--num-queries',
                       type=int,
                       default=100,
                       help='Number of queries to estimate for (default: 100)')
    parser.add_argument('--json',
                       action='store_true',
                       help='Output results in JSON format')
    
    args = parser.parse_args()
    
    data_dir = f"data/{args.dataset}"

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    # Calculate cost estimates
    extraction_cost = estimate_extraction_cost(data_dir, config)
    embedding_cost = estimate_embedding_cost(data_dir, config)
    query_cost = estimate_query_cost(config, args.num_queries)
    
    if args.json:
        # Output JSON format
        results = {
            "extraction": extraction_cost,
            "embedding": embedding_cost,
            "query": query_cost,
            "summary": {
                "setup_cost": extraction_cost['total_cost'] + embedding_cost['embedding_cost'],
                "per_query_cost": query_cost['total_cost'] / query_cost['num_queries']
            }
        }
        print(json.dumps(results, indent=2))
    else:
        # Output human-readable format
        print_cost_breakdown(extraction_cost, embedding_cost, query_cost)

if __name__ == "__main__":
    exit(main())