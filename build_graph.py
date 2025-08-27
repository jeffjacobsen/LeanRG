#!/usr/bin/env python3
"""
Enhanced build_graph.py with support for multiple API providers.
Supports Lambda Labs, OpenAI, and local LLM instances.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import field
import json
import os
import logging
import numpy as np
from openai import OpenAI, AsyncOpenAI
import tiktoken
from tqdm import tqdm
import yaml
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from utils.cluster_utils import Hierarchical_Clustering
from utils.file_utils import write_jsonl
from utils.database_utils_supabase import create_db_tables, insert_data_to_supabase
from utils.database_utils_mysql import create_db_table_mysql_config, insert_data_to_mysql_config
from utils.database_utils_qdrant import build_vector_search
from utils.llm_manager import create_llm_manager
from utils.embedding_manager import get_embedding_manager, create_pickleable_embedding_function
import requests
import multiprocessing

# Setup logging
# Setup logging for noisy libraries
for noisy in ["httpx", "openai", "urllib3", "aiohttp", "asyncio"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

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

class SyncLLMWrapper:
    """Thread-safe synchronous LLM wrapper for clustering (no async rate limiting)."""
    
    def __init__(self, async_llm_manager):
        # Extract config from async manager to create sync client
        self.api_provider = async_llm_manager.api_provider
        self.base_url = async_llm_manager.base_url
        self.api_key = async_llm_manager.api_key
        self.model = async_llm_manager.model
        self.max_retries = async_llm_manager.max_retries
        self.timeout = async_llm_manager.timeout
        
        # Create synchronous OpenAI client
        from openai import OpenAI
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
        
        # Enhanced thread-safe rate limiting for Groq's TPM limits
        import threading
        import time
        from collections import deque
        import tiktoken
        
        # Store time module for use in methods
        self.time = time
        
        # Token counting setup
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        self.rate_lock = threading.Lock()
        self.token_usage_history = deque()  # (timestamp, tokens_used)
        
        # Get rate limit from original config
        self.rate_limit_tpm = getattr(async_llm_manager, 'rate_limit_tpm', 150000)
        self.min_interval = 0.5  # Minimum 500ms between calls for clustering stability
        
        logger.info(f"SyncLLMWrapper rate limiting: {self.rate_limit_tpm} TPM, {self.min_interval}s min interval")
        
        logger.info(f"Created thread-safe sync LLM wrapper for {self.api_provider.upper()}")
        
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(self.tokenizer.encode(text))
    
    def _clean_old_usage_records(self, current_time: float):
        """Remove usage records older than 1 minute."""
        while self.token_usage_history and current_time - self.token_usage_history[0][0] > 60:
            self.token_usage_history.popleft()
    
    def _get_current_token_usage(self, current_time: float) -> int:
        """Get current token usage in the last minute."""
        self._clean_old_usage_records(current_time)
        return sum(tokens for _, tokens in self.token_usage_history)
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Thread-safe synchronous text generation with token-based rate limiting."""
        try:
            # Estimate tokens for this request
            estimated_tokens = self._estimate_tokens(prompt) + 1000  # Add estimated response tokens
            
            # Token-based rate limiting
            with self.rate_lock:
                current_time = self.time.time()
                
                # Check current usage
                current_usage = self._get_current_token_usage(current_time)
                
                # Check if we need to wait for token rate limit
                if current_usage + estimated_tokens > self.rate_limit_tpm:
                    # Wait until oldest record expires
                    if self.token_usage_history:
                        oldest_time = self.token_usage_history[0][0]
                        wait_time = 61 - (current_time - oldest_time)  # Wait for oldest to expire + buffer
                        if wait_time > 0:
                            logger.info(f"Rate limit protection: waiting {wait_time:.2f}s (current: {current_usage}, estimated: {estimated_tokens})")
                            # Release lock while sleeping
                    
                    # Re-acquire time after potential sleep
                    current_time = self.time.time()
                
                # Minimum interval between calls
                if hasattr(self, 'last_call_time'):
                    time_since_last = current_time - self.last_call_time
                    if time_since_last < self.min_interval:
                        sleep_time = self.min_interval - time_since_last
                        # Release lock while sleeping to prevent deadlock
                        
                self.last_call_time = current_time
            
            # Sleep outside of lock if needed
            if 'wait_time' in locals() and wait_time > 0:
                self.time.sleep(wait_time)
            if 'sleep_time' in locals() and sleep_time > 0:
                self.time.sleep(sleep_time)
            
            # Make synchronous API call
            messages = [{"role": "user", "content": prompt}]
            
            for attempt in range(self.max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        timeout=self.timeout,
                        **kwargs
                    )
                    
                    # Track actual token usage
                    actual_tokens = estimated_tokens  # fallback
                    if hasattr(response, 'usage') and response.usage:
                        actual_tokens = response.usage.total_tokens
                    
                    # Record token usage for rate limiting
                    with self.rate_lock:
                        self.token_usage_history.append((self.time.time(), actual_tokens))
                    
                    return response.choices[0].message.content
                    
                except Exception as e:
                    error_str = str(e)
                    
                    # Handle rate limit errors specifically
                    if "429" in error_str or "rate limit" in error_str.lower():
                        # Parse wait time from Groq error messages
                        import re
                        match = re.search(r'try again in (\\d+(?:\\.\\d+)?)s', error_str)
                        if match:
                            wait_time = float(match.group(1)) + 1  # Add buffer
                            logger.info(f"Rate limit hit, waiting {wait_time}s as suggested by API")
                            self.time.sleep(wait_time)
                            continue
                    
                    if attempt == self.max_retries - 1:
                        logger.error(f"All {self.max_retries} attempts failed: {error_str}")
                        return ""
                    
                    # Exponential backoff for other errors
                    self.time.sleep(2 ** attempt)
            
            return ""
            
        except Exception as e:
            logger.error(f"Error in synchronous LLM call: {e}")
            return ""

def setup_llm_manager(config_path: str):
    """Setup LLM manager with rate limiting based on configuration."""
    config = load_config(config_path)
    
    # Check if using API provider or local setup
    if 'llm_conf' in config and config['llm_conf'].get('api_provider') in ['lambda', 'openai', 'groq', 'cerebras']:
        llm_config = config['llm_conf']
        async_llm_manager = create_llm_manager(llm_config)
        
        if async_llm_manager:
            logger.info(f"Using {llm_config.get('api_provider', 'unknown').upper()} LLM Manager with rate limiting for clustering")
            # Wrap the async manager with a synchronous interface
            return SyncLLMWrapper(async_llm_manager)
        else:
            logger.error("Failed to create LLM manager")
            return None
    else:
        # Fallback to legacy local setup
        logger.warning("No API provider found in config")
        return None

def get_common_rag_res(WORKING_DIR):
    """Load and process entity and relation data."""
    entity_path = f"{WORKING_DIR}/entity.jsonl"
    relation_path = f"{WORKING_DIR}/relation.jsonl"
    
    e_dic = {}
    with open(entity_path, "r") as f:
        for xline in f:
            line = json.loads(xline)
            entity_name = str(line['entity_name'])
            description = line['description']
            source_id = line['source_id']
            if entity_name not in e_dic.keys():
                e_dic[entity_name] = dict(
                    entity_name=str(entity_name),
                    description=description,
                    source_id=source_id,
                    degree=0,
                )
            else:
                e_dic[entity_name]['description'] += "|Here is another description : " + description
                if e_dic[entity_name]['source_id'] != source_id:
                    e_dic[entity_name]['source_id'] += "|" + source_id
    
    r_dic = {}
    with open(relation_path, "r") as f:
        for xline in f:
            line = json.loads(xline)
            src_tgt = str(line['src_tgt'])
            tgt_src = str(line['tgt_src'])
            description = line['description']
            weight = 1
            source_id = line['source_id']
            r_dic[(src_tgt, tgt_src)] = {
                'src_tgt': str(src_tgt),
                'tgt_src': str(tgt_src),
                'description': description,
                'weight': weight,
                'source_id': source_id
            }
    
    return e_dic, r_dic

# Global embedding configuration (for multiprocessing)
EMBEDDING_CONFIG = None

# Removed duplicate EmbeddingFunction - now using shared one from utils.llm_manager

def setup_embedding_function(config_path: str):
    """Setup embedding function based on configuration with FastEmbed/OpenAI support."""
    global EMBEDDING_CONFIG
    
    config = load_config(config_path)
    
    # Use the new embedding manager for unified embedding support
    try:
        embedding_manager = get_embedding_manager(config)
        
        # Store for later use in worker processes
        EMBEDDING_CONFIG = config.get('embedding_conf', {})
        
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


def truncate_text(text, max_tokens=4096):
    """Truncate text to max tokens."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(tokens)
    return truncated_text

def embedding_init_worker(args):
    """Worker function for multiprocessing embedding initialization."""
    entities_batch, embedding_config = args
    
    # Create embedding function in worker process (using shared implementation)
    embedding_func = create_pickleable_embedding_function(embedding_config)
    
    # Process entities
    texts = [truncate_text(entity['description']) for entity in entities_batch]
    embeddings = embedding_func(texts)
    
    for i, entity in enumerate(entities_batch):
        entity['vector'] = np.array(embeddings[i])
    
    return entities_batch

def embedding_data(entity_results, embedding_func):
    """Generate embeddings for all entities."""
    global EMBEDDING_CONFIG
    
    entities = [v for k, v in entity_results.items()]
    entity_with_embeddings = []
    embeddings_batch_size = 64
    num_embeddings_batches = (len(entities) + embeddings_batch_size - 1) // embeddings_batch_size
    
    batches = [
        entities[i * embeddings_batch_size : min((i + 1) * embeddings_batch_size, len(entities))]
        for i in range(num_embeddings_batches)
    ]

    # Prepare arguments with embedding config for each batch
    worker_args = [(batch, EMBEDDING_CONFIG) for batch in batches]

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(embedding_init_worker, args) for args in worker_args]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            entity_with_embeddings.extend(result)

    for i in entity_with_embeddings:
        entity_name = i['entity_name']
        vector = i['vector']
        entity_results[entity_name]['vector'] = vector
    return entity_results

def hierarchical_clustering(global_config, config_path: str):
    """Perform hierarchical clustering."""

    #get entity and relation from entity.jsonl and relation.jsonl
    entity_results, relation_results = get_common_rag_res(global_config['working_dir'])
    
    #add embedding to entities
    embedding_func = global_config['embeddings_func']
    all_entities = embedding_data(entity_results, embedding_func)

    # At this point we have a standard Graph with entities and relations
    
    # Setup LLM manager
    llm_manager = setup_llm_manager(config_path)
    if llm_manager:
        # Use API-based LLM
        global_config['use_llm_func'] = llm_manager.generate_text
        logger.info("Using API-based LLM for clustering")
    else:
        logger.warning("No API found for clustering")
    
    # Perform clustering
    hierarchical_cluster = Hierarchical_Clustering()
    all_entities, generate_relations, community = hierarchical_cluster.perform_clustering(
        global_config=global_config,
        entities=all_entities,
        relations=relation_results,
        WORKING_DIR=global_config['working_dir'],
        max_workers=global_config['max_workers']
    )
    
    try:
        # Generate final embedding for root entity
        final_embedding = embedding_func([all_entities[-1]['description']])
        all_entities[-1]['vector'] = final_embedding[0]

        # Build vector search index using hybrid approach (Qdrant + Supabase)
        config = load_config(config_path)
        build_vector_search(all_entities, global_config['working_dir'], config)

    except Exception as e:
        logger.error(f"❌ Error in build_vector_search: {e}")
    
    # Save results
    # Note - all_entities.json is saved by hierarchical_clustering.perform_clustering function

    save_relation = [v for k, v in generate_relations.items()]
    save_community = [v for k, v in community.items()]
    write_jsonl(save_relation, f"{global_config['working_dir']}/generate_relations.json")
    write_jsonl(save_community, f"{global_config['working_dir']}/community.json")
    
    # Relational database storage
    config = load_config(config_path)
    database_backend = config.get('database', {}).get('backend', 'supabase')
    
    if database_backend == 'mysql':
        logger.info("🗄️ Using MySQL backend for relational data storage")
        create_db_table_mysql_config(global_config['working_dir'], config)
        insert_data_to_mysql_config(global_config['working_dir'], config)
    else:
        logger.info("🗄️ Using Supabase backend for relational data storage")
        create_db_tables({'config': config, 'working_dir': global_config['working_dir']})
        insert_data_to_supabase(global_config['working_dir'], config)


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    
    parser = argparse.ArgumentParser(description="Build hierarchical knowledge graph with API support")
    parser.add_argument("dataset_prefix", type=str, 
                       help="Dataset prefix (e.g., 'cs', 'legal', 'agriculture')")
    parser.add_argument("-c", "--config", type=str, default="config.yaml",
                       help="Configuration file path (default: config.yaml)")
    parser.add_argument("-w", "--workers", type=int, default=8,
                       help="Number of workers for processing")
    args = parser.parse_args()

    WORKING_DIR = f"data/{args.dataset_prefix}"
    CONFIG_PATH = args.config
    
    # Validate input files
    entity_path = os.path.join(WORKING_DIR, "entity.jsonl")
    relation_path = os.path.join(WORKING_DIR, "relation.jsonl")
    
    if not os.path.exists(entity_path):
        logger.error(f"Entity file not found: {entity_path}")
        sys.exit(1)
    
    if not os.path.exists(relation_path):
        logger.error(f"Relation file not found: {relation_path}")
        sys.exit(1)
    
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"Configuration file not found: {CONFIG_PATH}")
        sys.exit(1)
    
    # Setup embedding function (reuse same instance for consistency)
    embedding_func = setup_embedding_function(CONFIG_PATH)
    
    # Setup global configuration
    global_config = {
        'max_workers': args.workers,
        'working_dir': WORKING_DIR,
        'embeddings_func': embedding_func,  # Reuse same instance
        "special_community_report_llm_kwargs": field(
            default_factory=lambda: {"response_format": {"type": "json_object"}}
        )
    }
    
    logger.info(f"Starting hierarchical clustering for: {WORKING_DIR}")
    logger.info(f"Using configuration: {CONFIG_PATH}")
    logger.info(f"Workers: {args.workers}")
    
    hierarchical_clustering(global_config, CONFIG_PATH)
    
    logger.info("Hierarchical clustering completed successfully!")