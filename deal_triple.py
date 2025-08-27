#!/usr/bin/env python3
"""
Enhanced deal_triple.py for GraphExtraction with support for multiple API providers.
Processes GraphRAG extracted entities and relations with deduplication and summarization.
"""

import json
import os
import sys
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import tiktoken

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from prompt import PROMPTS
from utils.file_utils import write_jsonl
from utils.llm_manager import create_llm_manager
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_THRESHOLD = 100
tokenizer = tiktoken.get_encoding("cl100k_base")


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


async def summarize_entity_async(entity_name: str, description: str, llm_manager, summary_prompt: str, threshold: int) -> tuple:
    """
    Async version: Summarize entity description if it exceeds token threshold.
    
    Args:
        entity_name: Name of the entity
        description: Entity description to potentially summarize
        llm_manager: LLM manager instance
        summary_prompt: Prompt template for summarization
        threshold: Token threshold for summarization
        
    Returns:
        Tuple of (entity_name, processed_description)
    """
    tokens = len(tokenizer.encode(description))
    
    if tokens > threshold:
        try:
            exact_prompt = summary_prompt.format(entity_name=entity_name, description=description)
            response = await llm_manager.generate_text_async(exact_prompt)
            
            if response and response.strip():
                logger.debug(f"Summarized {entity_name}: {tokens} tokens → {len(tokenizer.encode(response))} tokens")
                return entity_name, response.strip()
            else:
                logger.warning(f"Empty summarization response for {entity_name}, using original")
                return entity_name, description
                
        except Exception as e:
            logger.error(f"Error summarizing {entity_name}: {e}")
            return entity_name, description
    
    return entity_name, description




async def deal_duplicate_entity_async(working_dir: str, llm_manager=None, threshold: int = DEFAULT_THRESHOLD):
    """
    Process and deduplicate entities from GraphRAG output with async LLM support.
    
    Args:
        working_dir: Directory containing input and output files
        llm_manager: LLM manager for summarization
        threshold: Token threshold for summarization
    """
    # No need to create directory - it should already exist with input files
    
    entity_path = os.path.join(working_dir, "entity-extract.jsonl")
    relation_path = os.path.join(working_dir, "relation-extract.jsonl")
    
    if not os.path.exists(entity_path):
        logger.error(f"Entity file not found: {entity_path}")
        raise FileNotFoundError(f"Entity file not found: {entity_path}")
    
    if not os.path.exists(relation_path):
        logger.error(f"Relation file not found: {relation_path}")
        raise FileNotFoundError(f"Relation file not found: {relation_path}")
    
    logger.info(f"Processing entities from: {entity_path}")
    logger.info(f"Processing relations from: {relation_path}")
    
    # Process entities with deduplication
    e_dic = {}
    summary_prompt = PROMPTS.get('summary_entities', 
                                "Summarize the following description for entity '{entity_name}': {description}")
    
    with open(entity_path, "r", encoding="utf-8") as f:
        for line_num, xline in enumerate(f):
            try:
                line = json.loads(xline)
                entity_name = str(line['entity_name']).replace('"', '')
                entity_type = line.get('entity_type', '').replace('"', '')
                description = line['description'].replace('"', '')
                source_id = line['source_id']
                
                if entity_name not in e_dic:
                    e_dic[entity_name] = {
                        'entity_name': str(entity_name),
                        'entity_type': entity_type,
                        'description': description,
                        'source_id': source_id,
                        'degree': 0,
                    }
                else:
                    # Merge descriptions and source IDs
                    e_dic[entity_name]['description'] += " | " + description
                    if e_dic[entity_name]['source_id'] != source_id:
                        e_dic[entity_name]['source_id'] += "|" + source_id
                        
            except Exception as e:
                logger.error(f"Error processing entity line {line_num + 1}: {e}")
                continue
    
    logger.info(f"Processed {len(e_dic)} unique entities")
    
    # Prepare entities for summarization
    all_entities = []
    to_summarize = []
    
    for entity_name, entity_data in e_dic.items():
        # Deduplicate source IDs
        entity_data['source_id'] = "|".join(set(entity_data['source_id'].split("|")))
        
        description = entity_data['description']
        tokens = len(tokenizer.encode(description))
        
        if tokens > threshold:
            to_summarize.append((entity_name, description, entity_data))
            logger.debug(f"Entity '{entity_name}' marked for summarization ({tokens} tokens)")
        else:
            all_entities.append(entity_data)
    
    logger.info(f"Entities to summarize: {len(to_summarize)}")
    logger.info(f"Entities not requiring summarization: {len(all_entities)}")
    
    # Summarize long descriptions
    if to_summarize and llm_manager:
        logger.info("Starting entity description summarization...")
        
        # Use asyncio for concurrent summarization
        import asyncio
        
        async def process_summaries():
            tasks = [
                summarize_entity_async(entity_name, desc, llm_manager, summary_prompt, threshold)
                for entity_name, desc, entity_data in to_summarize
            ]
            
            # Use asyncio.gather to maintain order
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                processed_results = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Error in summarization task for {to_summarize[i][0]}: {result}")
                        # Use original data as fallback
                        processed_results.append((to_summarize[i][:2], to_summarize[i][2]))
                    else:
                        processed_results.append((result, to_summarize[i][2]))
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Completed summarization: {i + 1}/{len(tasks)}")
                
                return processed_results
                
            except Exception as e:
                logger.error(f"Critical error in batch summarization: {e}")
                # Fallback: return original data for all entities
                return [(item[:2], item[2]) for item in to_summarize]
        
        # Run async summarization
        summarization_results = await process_summaries()
        
        # Process results
        for (entity_name, summarized_desc), entity_data in summarization_results:
            entity_data['description'] = summarized_desc
            all_entities.append(entity_data)
    else:
        # If no LLM manager available, add original entities
        for entity_name, desc, entity_data in to_summarize:
            all_entities.append(entity_data)
        logger.warning(f"No summarization available - would have summarized {len(to_summarize)} entities")
    
    # Save processed entities
    entity_output_path = os.path.join(working_dir, "entity.jsonl")
    write_jsonl(all_entities, entity_output_path)
    logger.info(f"Saved {len(all_entities)} entities to {entity_output_path}")
    
    # Process relations
    all_relations = []
    with open(relation_path, "r", encoding="utf-8") as f:
        for line_num, xline in enumerate(f):
            try:
                line = json.loads(xline)
                # Handle both list and dict formats
                if isinstance(line, list):
                    line = line[0]
                
                src_tgt = str(line['src_id']).replace('"', '')
                tgt_src = str(line['tgt_id']).replace('"', '')
                description = line['description'].replace('"', '')
                weight = 1
                source_id = line['source_id']
                
                all_relations.append({
                    'src_tgt': src_tgt,
                    'tgt_src': tgt_src,
                    'description': description,
                    'weight': weight,
                    'source_id': source_id
                })
                
            except Exception as e:
                logger.error(f"Error processing relation line {line_num + 1}: {e}")
                continue
    
    # Save processed relations
    relation_output_path = os.path.join(working_dir, "relation.jsonl")
    write_jsonl(all_relations, relation_output_path)
    logger.info(f"Saved {len(all_relations)} relations to {relation_output_path}")
    
    # Summary statistics
    total_entity_tokens = sum(len(tokenizer.encode(e['description'])) for e in all_entities)
    avg_entity_tokens = total_entity_tokens / len(all_entities) if all_entities else 0
    
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Entities: {len(all_entities)}")
    logger.info(f"Relations: {len(all_relations)}")
    logger.info(f"Average entity description length: {avg_entity_tokens:.1f} tokens")
    logger.info(f"Entities summarized: {len(to_summarize)}")


async def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(
        description="Process GraphExtraction entities and relations with API provider support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python deal_triple.py cs
  python deal_triple.py legal --threshold 75
  python deal_triple.py agriculture --no-summarize

Input/Output Structure:
  Working directory: data/{dataset_prefix}/
  Input files: entity-extract.jsonl, relation-extract.jsonl
  Output files: entity.jsonl, relation.jsonl
        """
    )
    
    parser.add_argument(
        "dataset_prefix",
        help="Dataset prefix (e.g., 'cs', 'legal', 'agriculture')"
    )
    
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Configuration file path (default: config.yaml)"
    )

    parser.add_argument(
        "--threshold", "-t",
        type=int,
        default=DEFAULT_THRESHOLD,
        help=f"Token threshold for description summarization (default: {DEFAULT_THRESHOLD})"
    )
    
    parser.add_argument(
        "--no-summarize",
        action="store_true",
        help="Skip description summarization (faster but may have long descriptions)"
    )
    
    
    args = parser.parse_args()
    
    working_dir = f"data/{args.dataset_prefix}"
    
    # Validate working directory
    if not os.path.exists(working_dir):
        logger.error(f"Working directory not found: {working_dir}")
        logger.error(f"Make sure you've run chunk.py {args.dataset_prefix} first")
        return 1
    
    # Load configuration if summarization is enabled
    llm_manager = None
    
    if not args.no_summarize:
        try:
            config = load_config(args.config)
            llm_conf = config.get("llm_conf", {})
            
            logger.info(f"Initializing LLM manager for summarization...")
            logger.info(f"API Provider: {llm_conf.get('api_provider', 'unknown')}")
            logger.info(f"Model: {llm_conf.get('llm_model', 'unknown')}")
            
            # Use async LLM manager
            llm_manager = create_llm_manager(llm_conf)
            logger.info("✅ LLM manager initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize LLM manager: {e}")
            logger.warning("Proceeding without summarization...")
    else:
        logger.info("Summarization disabled by --no-summarize flag")
    
    # Process entities and relations
    logger.info("=" * 60)
    logger.info("GraphExtraction Triple Processing")
    logger.info("=" * 60)
    logger.info(f"Dataset prefix: {args.dataset_prefix}")
    logger.info(f"Working directory: {working_dir}")
    logger.info(f"Summarization: {'disabled' if args.no_summarize else 'enabled'}")
    logger.info(f"Processing mode: {'async' if llm_manager else 'no-summarization'}")
    
    try:
        if llm_manager is None:
            # Process without summarization
            await deal_duplicate_entity_async(working_dir, None, args.threshold)
        else:
            # Process with async LLM summarization
            await deal_duplicate_entity_async(working_dir, llm_manager, args.threshold)
        
        logger.info("🎉 Entity and relation processing completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        return 1


if __name__ == "__main__":
    import asyncio
    import sys
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)