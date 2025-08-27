#!/usr/bin/env python3
"""
Enhanced query_graph.py with support for multiple API providers.
Supports Lambda Labs, OpenAI, and local LLM instances for both LLM and embedding operations.
"""

import os
# Suppress Pydantic plugin warnings
os.environ['PYDANTIC_DISABLE_PLUGINS'] = '1'

import argparse
from collections import Counter, defaultdict
from dataclasses import field
import json
import os
import time
import logging
import numpy as np
from openai import OpenAI
import tiktoken
from tqdm import tqdm
# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import yaml
from utils.file_and_instance_utils import InstanceManager
# Import LeanRAG algorithms from modernized database utilities
from utils.database_utils_supabase import (
    find_tree_root_supabase, find_path_supabase, search_nodes_link_supabase,
    search_nodes_supabase, search_community_supabase, get_text_units_supabase, search_chunks_supabase,
    search_multiple_relationships_supabase
)
from utils.database_utils_mysql import (
    find_tree_root_mysql, search_nodes_link_mysql, search_nodes_mysql,
    search_community_mysql, get_text_units_mysql
)
from utils.database_utils_qdrant import search_vector_search_qdrant
from utils.llm_manager import create_sync_llm_manager, create_embedding_manager
from utils.embedding_manager import get_embedding_manager

# Database backend availability flags
SUPABASE_AVAILABLE = True  # We have implemented these functions
QDRANT_AVAILABLE = True    # We have implemented these functions
from prompt import GRAPH_FIELD_SEP, PROMPTS
from itertools import combinations
import sys
from pathlib import Path

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise

# Removed duplicate APILLMManager - now using shared one from utils.llm_manager

# Removed duplicate APIEmbeddingManager - now using shared one from utils.llm_manager

def setup_llm_manager(config_path: str):
    """Setup LLM manager based on configuration."""
    config = load_config(config_path)
    
    # Check if using API provider or local setup
    if 'llm_conf' in config and config['llm_conf'].get('api_provider') in ['lambda', 'openai', 'groq', 'cerebras', 'local']:
        llm_config = config['llm_conf']
        return create_sync_llm_manager(llm_config)
    else:
        # Fallback to legacy local setup
        logger.warning("No API provider found in config, falling back to local InstanceManager setup")
        return None

def setup_embedding_manager(config_path: str):
    """Setup embedding manager based on configuration."""
    config = load_config(config_path)
    
    # Use the new unified embedding manager
    try:
        embedding_manager = get_embedding_manager(config)
        
        # Create function wrapper that works with existing code
        def embedding_func(texts):
            """Wrapper function for embedding manager."""
            if isinstance(texts, str):
                return [embedding_manager.embed_single(texts)]
            else:
                embeddings = embedding_manager.embed_texts(texts)
                return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        
        # Log model information
        model_info = embedding_manager.get_model_info()
        logger.info(f"✅ Initialized {model_info['provider'].upper()} embedding function")
        logger.info(f"📊 Model: {model_info.get('name', 'unknown')} ({model_info['dimensions']} dims)")
        
        return embedding_func
        
    except Exception as e:
        logger.error(f"❌ Failed to setup new embedding function: {e}")
        logger.warning("🔄 Falling back to legacy embedding function...")
        
        # Fallback to legacy implementation
        if 'embedding_conf' in config:
            embedding_config = config['embedding_conf']
            return create_embedding_manager(embedding_config)
        elif 'glm' in config:
            embedding_config = {
                'api_provider': 'local',
                'model': config['glm']['model'],
                'base_url': config['glm']['base_url'],
                'api_key': config['glm']['model']
            }
            return create_embedding_manager(embedding_config)
        else:
            embedding_config = {
                'api_provider': 'local',
                'model': 'bge_m3',
                'base_url': 'http://localhost:8000/v1',
                'api_key': 'bge_m3'
            }
            return create_embedding_manager(embedding_config)

# Tokenizer for text truncation
tokenizer = tiktoken.get_encoding("cl100k_base")

def truncate_text(text, max_tokens=4096):
    """Truncate text to maximum number of tokens."""
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(tokens)
    return truncated_text

def get_reasoning_chain(global_config, entities_set):
    """Generate reasoning chains between entities."""
    maybe_edges = list(combinations(entities_set, 2))
    reasoning_path = []
    reasoning_path_information = []
    working_dir = global_config['working_dir']
    information_record = []
    
    # Determine database backend
    database_backend = global_config.get('database', {}).get('backend', 'supabase')
    
    # Debug logging
    # logger.info(f"🔍 get_reasoning_chain called with {len(entities_set)} entities: {entities_set}")
    # logger.info(f"🔧 Database backend: {database_backend}")
    # logger.info(f"📊 Generated {len(maybe_edges)} entity pairs to process")
    
    # Collect all entity pairs from all reasoning paths
    all_entity_pairs = []
    
    for edge in maybe_edges:
        a_path = []
        b_path = []
        node1 = edge[0]
        node2 = edge[1]
        
        # Use appropriate tree root function based on backend
        if database_backend == 'mysql':
            node1_tree = find_tree_root_mysql(node1, working_dir, global_config)
            node2_tree = find_tree_root_mysql(node2, working_dir, global_config)
        else:
            node1_tree = find_tree_root_supabase(node1, working_dir, global_config)
            node2_tree = find_tree_root_supabase(node2, working_dir, global_config)
        
        for index, (i, j) in enumerate(zip(node1_tree, node2_tree)):
            if i == j:
                a_path.append(i)
                break
            if i in b_path or j in a_path:
                break
            if i != j:
                a_path.append(i)
                b_path.append(j)
        
        reasoning_path.append(a_path + [b_path[len(b_path) - 1 - i] for i in range(len(b_path))])
        a_path = list(set(a_path))
        b_path = list(set(b_path))
        
        # Collect entity pairs from this reasoning path
        for maybe_edge in list(combinations(a_path + b_path, 2)):
            if maybe_edge[0] != maybe_edge[1]:
                all_entity_pairs.append(maybe_edge)
    
    # Process relationship searches based on backend
    if all_entity_pairs:
        # Remove duplicates while preserving order
        unique_pairs = list(dict.fromkeys(all_entity_pairs))
        
        if database_backend == 'mysql':
            # Use individual MySQL calls (they're fast enough locally)
            for maybe_edge in unique_pairs:
                information = search_nodes_link_mysql(maybe_edge[0], maybe_edge[1], working_dir, global_config)
                if information is not None:
                    information_record.append(information)
                    reasoning_path_information.append([maybe_edge[0], maybe_edge[1], information[2]])
        else:
            # Use batch Supabase processing for better network efficiency
            relationship_results = search_multiple_relationships_supabase(unique_pairs, working_dir, global_config)
            for maybe_edge in unique_pairs:
                information = relationship_results.get(maybe_edge)
                if information is not None:
                    information_record.append(information)
                    reasoning_path_information.append([maybe_edge[0], maybe_edge[1], information[2]])
    
    temp_relations_information = list(set([information[2] for information in reasoning_path_information]))
    reasoning_path_information_description = "\\n".join(temp_relations_information)
    
    return reasoning_path, reasoning_path_information_description

def get_entity_description(global_config, entities_set, mode=0):
    """Format entity descriptions for prompt."""
    columns = ['entity_name', 'parent', 'description']
    entity_descriptions = "\\t\\t".join(columns) + "\\n"
    entity_descriptions += "\\n".join([
        information[0] + "\\t\\t" + information[1] + "\\t\\t" + information[2] 
        for information in entities_set
    ])
    return entity_descriptions

def get_aggregation_description(global_config, reasoning_path, if_findings=False):
    """Get aggregation descriptions from community data."""
    aggregation_results = []
    
    # Determine database backend
    database_backend = global_config.get('database', {}).get('backend', 'supabase')
    
    communities = set([community for each_path in reasoning_path for community in each_path])
    for community in communities:
        if database_backend == 'mysql':
            temp = search_community_mysql(community, global_config['working_dir'], global_config)
        else:
            temp = search_community_supabase(community, global_config['working_dir'], global_config)
        
        # Handle None values
        if temp is None or temp == "":
            continue
        aggregation_results.append(temp)
    
    if if_findings:
        columns = ['entity_name', 'entity_description', 'findings']
        aggregation_descriptions = "\\t\\t".join(columns) + "\\n"
        aggregation_descriptions += "\\n".join([
            information[0] + "\\t\\t" + str(information[1]) + "\\t\\t" + information[2] 
            for information in aggregation_results if information is not None and len(information) >= 3
        ])
    else:
        columns = ['entity_name', 'entity_description']
        aggregation_descriptions = "\\t\\t".join(columns) + "\\n"
        aggregation_descriptions += "\\n".join([
            information[0] + "\\t\\t" + str(information[1]) 
            for information in aggregation_results if information is not None and len(information) >= 2
        ])
    
    return aggregation_descriptions, communities

def query_graph(global_config, db, query):
    """
    Main query function with API support.
    
    Args:
        global_config: Configuration dictionary
        db: Database connection
        query: Query string
        
    Returns:
        Tuple of (context, response)
    """
    use_llm_func = global_config["use_llm_func"]
    embedding_func = global_config["embeddings_func"]
    
    b = time.time()
    level_mode = global_config['level_mode']
    topk = global_config['topk']
    
    # Get query embedding and search for relevant entities
    query_embedding = embedding_func([query])
    logger.info("🔧 Using Qdrant for vector search (Supabase + Qdrant architecture)")
    entity_results = search_vector_search_qdrant(
        global_config['working_dir'], 
        query_embedding[0], 
        global_config,
        topk=topk, 
        level_mode=level_mode
    )
    logger.info("Qdrant returned entity_results")

    v = time.time()
    res_entity = [i[0] for i in entity_results]
    chunks = [i[-1] for i in entity_results]
    
    # Build context information
    entity_descriptions = get_entity_description(global_config, entity_results)
    logger.info("returned entity_descriptions")
    reasoning_path, reasoning_path_information_description = get_reasoning_chain(global_config, res_entity)
    logger.info("returned chain")
    aggregation_descriptions, aggregation = get_aggregation_description(global_config, reasoning_path)
    
    # Get text units based on database backend
    database_backend = global_config.get('database', {}).get('backend', 'supabase')
    if database_backend == 'mysql':
        text_units = get_text_units_mysql(global_config['working_dir'], chunks, global_config, k=5)
    else:
        text_units = get_text_units_supabase(global_config['working_dir'], chunks, global_config, k=5)
    logger.info(f"test units {text_units}")
    
    # Format context
    describe = f"""
entity_information:
{entity_descriptions}
aggregation_entity_information:
{aggregation_descriptions}
reasoning_path_information:
{reasoning_path_information_description}
text_units:
{text_units}
"""
    
    e = time.time()
    
    # Generate response using LLM
    sys_prompt = PROMPTS["rag_response"].format(context_data=describe)
    response = use_llm_func(query, system_prompt=sys_prompt)
    
    g = time.time()
    
    # Print timing information
    print(f"embedding time: {v-b:.2f}s")
    print(f"query time: {e-v:.2f}s")
    print(f"response time: {g-e:.2f}s")
    
    return describe, response

def main():
    """Main function with argument parsing and API setup."""
    parser = argparse.ArgumentParser(description="Query knowledge graph with API support")
    parser.add_argument("dataset_prefix", type=str,
                       help="Dataset prefix (e.g., 'cs', 'legal', 'agriculture')")
    parser.add_argument("-q", "--query", type=str, required=True,
                       help="Query string")
    parser.add_argument("-c", "--config", type=str, default="config.yaml",
                       help="Configuration file path (default: config.yaml)")
    parser.add_argument("--topk", type=int, default=10,
                       help="Number of top entities to retrieve")
    parser.add_argument("--level-mode", type=int, default=2,
                       help="Level mode: 0=original nodes, 1=aggregated nodes, 2=all nodes (recommended)")
    
    args = parser.parse_args()
    
    # Construct working directory from dataset prefix
    working_dir = f"data/{args.dataset_prefix}"
    
    # Validate paths
    if not os.path.exists(working_dir):
        logger.error(f"Working directory not found: {working_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Setup API managers
    llm_manager = setup_llm_manager(args.config)
    embedding_manager = setup_embedding_manager(args.config)
    
    if llm_manager is None:
        logger.error("Failed to setup LLM manager")
        sys.exit(1)
    
    # Load configuration for database settings
    config = load_config(args.config)
    
    # Setup global configuration
    global_config = {
        'working_dir': working_dir,
        'embeddings_func': embedding_manager,
        'use_llm_func': llm_manager.generate_text,
        'topk': args.topk,
        'level_mode': args.level_mode,
        'database': config.get('database', {'backend': 'mysql'}),
        'supabase_conf': config.get('supabase_conf', {})
    }
    
    # Database connection handling
    try:
        database_backend = global_config['database'].get('backend', 'supabase')
        if database_backend == 'mysql':
            logger.info("🗄️ Using MySQL + Qdrant backend")
        else:
            logger.info("🗄️ Using Supabase + Qdrant backend")
            if not SUPABASE_AVAILABLE or not QDRANT_AVAILABLE:
                logger.error("❌ Supabase + Qdrant utilities not available")
                sys.exit(1)
        
        logger.info(f"Querying: {args.query}")
        logger.info(f"Working directory: {working_dir}")
        logger.info(f"Top-k: {args.topk}")
        logger.info(f"Level mode: {args.level_mode}")
        logger.info(f"Database backend: {database_backend}")
        
        # Execute query
        context, response = query_graph(global_config, None, args.query)
        
        # Print results
        print("\\n" + "="*80)
        print("CONTEXT:")
        print("="*80)
        print(context)
        print("\\n" + "="*80)
        print("RESPONSE:")
        print("="*80)
        print(response)
        
        # Print statistics
        print("\\n" + "="*80)
        print("STATISTICS:")
        print("="*80)
        print(f"LLM requests: {llm_manager.total_requests}")
        print(f"LLM tokens: {llm_manager.total_tokens}")
        
        # Database connections handled automatically by Supabase and Qdrant clients
        
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()