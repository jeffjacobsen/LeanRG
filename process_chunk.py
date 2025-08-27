#!/usr/bin/env python3
"""
Enhanced GraphExtraction chunk processing with support for multiple API providers.
Supports Lambda Labs, OpenAI, and local LLM instances.
"""

import os
import json
import sys
import asyncio
import argparse
import yaml
import copy
import time
from pathlib import Path
from collections import defaultdict
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.extraction_utils import (
    split_string_by_multi_markers, _handle_single_entity_extraction,
    _handle_single_relationship_extraction, pack_user_ass_to_openai_messages
)
from utils.llm_manager import create_llm_manager
from utils.file_utils import write_jsonl
from prompt import PROMPTS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


def get_chunks(chunk_file: str) -> dict:
    """Load document chunks from file."""
    logger.info(f"Loading chunks from: {chunk_file}")
    
    doc_name = os.path.basename(chunk_file).rsplit(".", 1)[0]
    
    try:
        with open(chunk_file, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        
        chunks = {item["hash_code"]: item["text"] for item in corpus}
        logger.info(f"Loaded {len(chunks)} chunks from {doc_name}")
        return chunks
        
    except Exception as e:
        logger.error(f"Error loading chunks: {e}")
        raise


async def extract_entities_and_relations(
    chunks: dict, 
    llm_manager,
    output_dir: str,
    config: dict
) -> None:
    """
    Extract entities and relations from chunks using LLM.
    
    Args:
        chunks: Dictionary of chunk_id -> text
        llm_manager: LLM manager instance
        output_dir: Output directory for results
        config: Configuration dictionary
    """
    
    logger.info("Starting entity and relation extraction...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Progress tracking
    total_chunks = len(chunks)
    processed_chunks = 0
    extracted_entities = 0
    extracted_relations = 0
    
    ordered_chunks = list(chunks.items())
    
    # Extract entities first
    logger.info("Phase 1: Extracting entities...")
    entity_results = await _extract_entities(ordered_chunks, llm_manager, config)
    
    # Process entity results
    all_entities = {}
    context_entities = {}
    
    for i, (chunk_key, _) in enumerate(ordered_chunks):
        entities, _ = entity_results[i]
        for entity_name, entity_list in entities.items():
            if entity_list:  # Take first occurrence
                all_entities[entity_name] = entity_list[0]
        
        # Store entities for this chunk (for relation extraction)
        context_entities[chunk_key] = list(entities.keys())
    
    logger.info(f"Extracted {len(all_entities)} unique entities")
    
    # Extract relations second
    logger.info("Phase 2: Extracting relations...")
    relation_results = await _extract_relations(ordered_chunks, context_entities, llm_manager, config)
    
    # Process relation results
    all_relations = {}
    for i, (chunk_key, _) in enumerate(ordered_chunks):
        _, relations = relation_results[i]
        for relation_key, relation_list in relations.items():
            if relation_list:  # Take first occurrence
                all_relations[relation_key] = relation_list[0]
    
    logger.info(f"Extracted {len(all_relations)} unique relations")
    
    # Save results
    await _save_results(all_entities, all_relations, output_dir)
    
    # Print statistics
    stats = llm_manager.get_stats()
    logger.info("=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total entities: {len(all_entities)}")
    logger.info(f"Total relations: {len(all_relations)}")
    logger.info(f"API Provider: {stats.get('api_provider', 'unknown')}")
    logger.info(f"Model: {stats.get('model', 'unknown')}")
    logger.info(f"Total API calls: {stats.get('total_requests', 0)}")
    logger.info(f"Total tokens: {stats.get('total_tokens', 0)}")


async def _extract_entities(ordered_chunks, llm_manager, config):
    """Extract entities from all chunks."""
    
    entity_extract_max_gleaning = config.get("task_conf", {}).get("entity_extract_max_gleaning", 1)
    
    async def _process_single_content_entity(chunk_key_dp):
        chunk_key, content = chunk_key_dp
        
        # Prepare context for entity extraction
        context_base_entity = {
            'tuple_delimiter': PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            'record_delimiter': PROMPTS["DEFAULT_RECORD_DELIMITER"],
            'completion_delimiter': PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            'entity_types': ",".join(PROMPTS["META_ENTITY_TYPES"])
        }
        
        # Initial extraction
        entity_extract_prompt = PROMPTS["entity_extraction"]
        hint_prompt = entity_extract_prompt.format(**context_base_entity, input_text=content)
        
        final_result = await llm_manager.generate_text_async(hint_prompt)
        
        # Iterative gleaning
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        
        for glean_index in range(entity_extract_max_gleaning):
            continue_prompt = PROMPTS["entiti_continue_extraction"]
            glean_result = await llm_manager.generate_text_async(
                continue_prompt, 
                history_messages=history
            )
            
            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            
            if glean_index == entity_extract_max_gleaning - 1:
                break
            
            # Check if we should continue
            if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]
            if_loop_result = await llm_manager.generate_text_async(
                if_loop_prompt,
                history_messages=history
            )
            
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break
        
        # Process results
        return await _process_extraction_results(
            final_result, chunk_key, context_base_entity
        )
    
    # Process all chunks concurrently
    results = await asyncio.gather(
        *[_process_single_content_entity(chunk) for chunk in ordered_chunks]
    )
    
    return results


async def _extract_relations(ordered_chunks, context_entities, llm_manager, config):
    """Extract relations from all chunks using pre-extracted entities."""
    
    entity_extract_max_gleaning = config.get("task_conf", {}).get("entity_extract_max_gleaning", 1)
    
    async def _process_single_content_relation(chunk_key_dp):
        chunk_key, content = chunk_key_dp
        
        # Get entities for this chunk
        entities = context_entities.get(chunk_key, [])
        if not entities:
            return {}, {}
        
        # Prepare context for relation extraction
        context_base_relation = {
            'tuple_delimiter': PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            'record_delimiter': PROMPTS["DEFAULT_RECORD_DELIMITER"],
            'completion_delimiter': PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            'entities': ",".join(entities)
        }
        
        # Initial extraction
        relation_extract_prompt = PROMPTS["relation_extraction"]
        hint_prompt = relation_extract_prompt.format(**context_base_relation, input_text=content)
        
        final_result = await llm_manager.generate_text_async(hint_prompt)
        
        # Iterative gleaning (if enabled)
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        
        for glean_index in range(entity_extract_max_gleaning):
            continue_prompt = PROMPTS["entiti_continue_extraction"]
            glean_result = await llm_manager.generate_text_async(
                continue_prompt,
                history_messages=history
            )
            
            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            
            if glean_index == entity_extract_max_gleaning - 1:
                break
            
            # Check if we should continue
            if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]
            if_loop_result = await llm_manager.generate_text_async(
                if_loop_prompt,
                history_messages=history
            )
            
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break
        
        # Process results
        return await _process_extraction_results(
            final_result, chunk_key, context_base_relation
        )
    
    # Process all chunks concurrently
    results = await asyncio.gather(
        *[_process_single_content_relation(chunk) for chunk in ordered_chunks]
    )
    
    return results


async def _process_extraction_results(final_result, chunk_key, context_base):
    """Process extraction results to get entities and relations."""
    
    records = split_string_by_multi_markers(
        final_result,
        [context_base["record_delimiter"], context_base["completion_delimiter"]],
    )
    
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    
    for record in records:
        import re
        record_match = re.search(r"\((.*)\)", record)
        if record_match is None:
            continue
        
        record = record_match.group(1)
        record_attributes = split_string_by_multi_markers(
            record, [context_base["tuple_delimiter"]]
        )
        
        # Try to extract entity
        entity = await _handle_single_entity_extraction(record_attributes, chunk_key)
        if entity is not None:
            maybe_nodes[entity["entity_name"]].append(entity)
            continue
        
        # Try to extract relation
        relation = await _handle_single_relationship_extraction(record_attributes, chunk_key)
        if relation is not None:
            relation_key = (relation["src_id"], relation["tgt_id"])
            maybe_edges[relation_key].append(relation)
    
    return dict(maybe_nodes), dict(maybe_edges)


async def _save_results(all_entities, all_relations, output_dir):
    """Save extraction results to files."""
    
    # Prepare entity data
    save_entities = []
    for entity_name, entity_data in all_entities.items():
        entity_copy = copy.deepcopy(entity_data)
        # Remove embedding if present
        entity_copy.pop('embedding', None)
        save_entities.append(entity_copy)
    
    # Prepare relation data
    save_relations = []
    for relation_key, relation_data in all_relations.items():
        if isinstance(relation_data, list):
            relation_data = relation_data[0]  # Take first if it's a list
        save_relations.append(relation_data)
    
    # Write to files
    entity_file = os.path.join(output_dir, "entity-extract.jsonl")
    relation_file = os.path.join(output_dir, "relation-extract.jsonl")
    
    write_jsonl(save_entities, entity_file)
    write_jsonl(save_relations, relation_file)
    
    logger.info(f"Saved {len(save_entities)} entities to {entity_file}")
    logger.info(f"Saved {len(save_relations)} relations to {relation_file}")


async def main():
    """Main execution function."""
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="GraphExtraction entity and relation extraction")
    parser.add_argument(
        "dataset_prefix",
        help="Dataset prefix (e.g., 'cs', 'legal', 'agriculture')"
    )
    parser.add_argument(
        "--working-dir", "-w",
        default="data",
        help="Working directory (default: data)"
    )

    args = parser.parse_args()
    
    # Load configuration

    config = load_config("config.yaml")
        # Extract configuration sections
    llm_conf = config.get("llm_conf", {})
    task_conf = config.get("graphextraction_conf", config.get("task_conf", {}))  # Support both new and legacy format
    
    logger.info("=" * 60)
    logger.info("GraphExtraction with API Support")
    logger.info("=" * 60)
    logger.info(f"Dataset prefix: {args.dataset_prefix}")
    logger.info(f"Configuration file: {args.config}")
    logger.info(f"API Provider: {llm_conf.get('api_provider', 'unknown')}")
    logger.info(f"Model: {llm_conf.get('llm_model', 'unknown')}")
    logger.info(f"Base URL: {llm_conf.get('llm_url', 'unknown')}")
    
    # Initialize LLM manager
    try:
        llm_manager = create_llm_manager(llm_conf)
        logger.info("✅ LLM Manager initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize LLM Manager: {e}")
        return 1
    
    # Load chunks using dataset prefix
    chunk_file = f"{args.working_dir}/{args.dataset_prefix}/chunk.json"
    try:
        chunks = get_chunks(chunk_file)
        logger.info(f"Loaded {len(chunks)} chunks from {chunk_file}")
    except Exception as e:
        logger.error(f"Failed to load chunks from {chunk_file}: {e}")
        return 1
    
    # Extract entities and relations

    output_dir = f"{args.working_dir}/{args.dataset_prefix}"
    
    start_time = time.time()
    try:
        await extract_entities_and_relations(chunks, llm_manager, output_dir, config)
        end_time = time.time()
        
        logger.info(f"⏱️  Total processing time: {end_time - start_time:.2f} seconds")
        logger.info("🎉 Extraction completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Extraction failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)