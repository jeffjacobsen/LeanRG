#!/usr/bin/env python3
"""
Supabase Database Utilities for LeanRAG

This module provides Supabase-based replacements for MySQL + ChromaDB functionality,
unifying vector and relational operations in a single PostgreSQL + pgvector database.

Key Features:
- Vector similarity search with pgvector
- Relational queries for entity relationships
- Hierarchical path finding with recursive CTEs
- Batch operations for performance
- Project isolation via metadata
- Error handling and retry logic
"""

import json
import os
import numpy as np
from supabase import create_client, Client
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import Counter
import logging
from pathlib import Path
import time

# Global connection cache to avoid repeated Supabase client initialization
_supabase_manager_cache = None
import hashlib

# Setup logging
logger = logging.getLogger(__name__)

def get_project_table_name(base_table: str, project_name: str) -> str:
    """Get project-specific table name."""
    # Sanitize project name for table naming (replace special chars with underscores)
    sanitized = "".join(c if c.isalnum() else '_' for c in project_name).lower().strip('_')
    return f"{sanitized}_{base_table}"

def get_all_records_paginated(supabase_client, table_name, select_columns='*', filter_conditions=None, batch_size=1000):
    """
    Fetch all records from a Supabase table with proper pagination handling.
    
    Args:
        supabase_client: Supabase client instance
        table_name: Name of the table to query
        select_columns: Columns to select (default: '*')
        filter_conditions: Function that applies filters to the query
        batch_size: Number of records per batch (default: 1000)
    
    Returns:
        List of all records matching the criteria
    """
    all_records = []
    offset = 0
    
    while True:
        query = supabase_client.table(table_name).select(select_columns).range(offset, offset + batch_size - 1)
        
        if filter_conditions:
            query = filter_conditions(query)
        
        result = query.execute()
        
        if not result.data:
            break
            
        all_records.extend(result.data)
        
        # If we got fewer records than batch_size, we've reached the end
        if len(result.data) < batch_size:
            break
            
        offset += batch_size
    
    return all_records

class SupabaseManager:
    """Supabase database manager for LeanRAG operations."""
    
    def __init__(self, supabase_url, supabase_key, max_connections, timeout):
        """Initialize Supabase client with connection pooling."""
        try:
            self.supabase: Client = create_client(supabase_url, supabase_key)
            self.url = supabase_url
            self.max_connections = max_connections
            self.timeout = timeout
            logger.info(f"✅ Supabase client initialized: {supabase_url}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase client: {e}")
            raise
        
    def test_connection(self) -> bool:
        """Test Supabase connection and verify schema."""
        try:
            # Test basic connectivity by checking if we can call a simple RPC function
            # This avoids assuming specific tables exist
            result = self.supabase.rpc('get_project_table_names', {'project_name': 'test'}).execute()
            logger.info("✅ Supabase connection test passed")
            return True
        except Exception as e:
            logger.error(f"❌ Supabase connection test failed: {e}")
            return False
    
    def get_table_stats(self, project_name: str = None) -> Dict[str, int]:
        """Get table statistics for monitoring."""
        stats = {}
        tables = ['entities', 'relations', 'communities', 'text_units']
        
        for table in tables:
            try:
                query = self.supabase.table(table).select('id', count='exact')
                if project_name:
                    query = query.eq('metadata->project', project_name)
                result = query.execute()
                stats[table] = result.count or 0
            except Exception as e:
                logger.warning(f"Failed to get stats for {table}: {e}")
                stats[table] = -1
        
        return stats
    
    def create_project_tables(self, project_name: str) -> bool:
        """Create project-specific tables in Supabase."""
        try:
            if not self.supabase:
                logger.error("❌ Supabase client not available")
                return False
            
            # Sanitize project name
            sanitized = "".join(c if c.isalnum() else '_' for c in project_name).lower().strip('_')
            
            # Try using stored procedure first
            try:
                result = self.supabase.rpc('create_project_tables', {'project_name': project_name}).execute()
                if result.data:
                    logger.info(f"✅ {result.data}")
                    return True
            except Exception as e:
                logger.warning(f"⚠️ Stored procedure failed ({e}), trying direct table creation...")
            
            # Fallback: Try to create tables directly using table names
            tables = [
                f"{sanitized}_entities", 
                f"{sanitized}_relations", 
                f"{sanitized}_communities", 
                f"{sanitized}_text_units"
            ]
            
            # Check if tables already exist by querying them
            existing_tables = 0
            for table_name in tables:
                try:
                    result = self.supabase.table(table_name).select('*').limit(1).execute()
                    existing_tables += 1
                    logger.info(f"✅ Table {table_name} already exists")
                except Exception:
                    logger.warning(f"⚠️ Table {table_name} does not exist yet")
            
            if existing_tables == len(tables):
                logger.info(f"✅ All project tables exist for {project_name}")
                return True
            else:
                logger.error(f"❌ Missing tables for project {project_name}. Please create them manually or apply permission fixes.")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error creating project tables for {project_name}: {e}")
            return False
 
def get_supabase_manager(config: dict) -> SupabaseManager:
    """Create SupabaseManager from configuration with caching to avoid repeated initialization."""
    global _supabase_manager_cache
    
    supabase_config = config.get('supabase_conf')
    supabase_url = supabase_config.get('url') or os.getenv('SUPABASE_URL')
    supabase_key = supabase_config.get('key') or os.getenv('SUPABASE_ANON_KEY')
    
    if not supabase_url or not supabase_key:
        raise ValueError("Supabase URL and key must be provided in config or environment variables")
    
    # Create cache key based on connection parameters
    cache_key = f"{supabase_url}:{supabase_key[:10]}"  # Only first 10 chars of key for security
    
    # Return cached manager if exists and matches current config
    if _supabase_manager_cache is not None:
        cached_key = f"{_supabase_manager_cache.url}:{supabase_key[:10]}"
        if cached_key == cache_key:
            return _supabase_manager_cache
    
    # Create new manager and cache it
    max_connections = supabase_config.get('max_connections', 10)
    timeout = supabase_config.get('timeout', 30)
    
    _supabase_manager_cache = SupabaseManager(supabase_url, supabase_key, max_connections, timeout)
    return _supabase_manager_cache

def search_nodes_link_supabase(entity1: str, entity2: str, working_dir: str, config: dict) -> Optional[Tuple[str, str, str]]:
    """
    Search for relationship between two entities using project-specific relations table.
    
    Args:
        entity1: First entity name
        entity2: Second entity name  
        working_dir: Working directory for project identification
        config: Configuration dictionary
        
    Returns:
        Tuple of (src_tgt, tgt_src, description) or None if no relationship found
    """
    supabase_manager = get_supabase_manager(config)
    project_name = os.path.basename(working_dir)
    relations_table = get_project_table_name('relations', project_name)
    
    try:
        # Replicate original MySQL pattern: try direct match first, then reverse
        # Single query with OR condition to minimize API calls
        result = supabase_manager.supabase.table(relations_table).select(
            'src_tgt, tgt_src, description'
        ).or_(
            f'and(src_tgt.eq.{entity1},tgt_src.eq.{entity2}),and(src_tgt.eq.{entity2},tgt_src.eq.{entity1})'
        ).limit(1).execute()
        
        if result.data and len(result.data) > 0:
            row = result.data[0]
            return (row['src_tgt'], row['tgt_src'], row['description'])
        
        return None
            
    except Exception as e:
        logger.error(f"Error finding relationship between {entity1} and {entity2}: {e}")
        return None

def search_multiple_relationships_supabase(entity_pairs: List[tuple], working_dir: str, config: dict) -> Dict[tuple, Optional[Tuple[str, str, str]]]:
    """
    Batch search for multiple entity relationships to reduce API calls.
    
    Args:
        entity_pairs: List of (entity1, entity2) tuples to search
        working_dir: Working directory for project identification  
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping (entity1, entity2) -> relationship tuple or None
    """
    if not entity_pairs:
        return {}
        
    supabase_manager = get_supabase_manager(config)
    project_name = os.path.basename(working_dir)
    relations_table = get_project_table_name('relations', project_name)
    
    try:
        # Build a single query for all entity pairs
        or_conditions = []
        for entity1, entity2 in entity_pairs:
            or_conditions.append(f'and(src_tgt.eq.{entity1},tgt_src.eq.{entity2})')
            or_conditions.append(f'and(src_tgt.eq.{entity2},tgt_src.eq.{entity1})')
        
        # Single batch query for all relationships
        result = supabase_manager.supabase.table(relations_table).select(
            'src_tgt, tgt_src, description'
        ).or_(','.join(or_conditions)).execute()
        
        # Process results into lookup dictionary
        relationship_map = {}
        for entity1, entity2 in entity_pairs:
            relationship_map[(entity1, entity2)] = None
        
        for row in result.data:
            src, tgt = row['src_tgt'], row['tgt_src']
            relationship = (row['src_tgt'], row['tgt_src'], row['description'])
            
            # Map to original entity pairs (both directions)
            for entity1, entity2 in entity_pairs:
                if (src == entity1 and tgt == entity2) or (src == entity2 and tgt == entity1):
                    relationship_map[(entity1, entity2)] = relationship
                    break
        
        return relationship_map
        
    except Exception as e:
        logger.error(f"Error in batch relationship search: {e}")
        return {pair: None for pair in entity_pairs}

def search_nodes_supabase(entity_name: str, working_dir: str, config: dict) -> List[Tuple[str, str, str]]:
    """
    Search for entity information.
    
    Args:
        entity_name: Entity name to search for
        working_dir: Working directory for project identification
        config: Configuration dictionary
        
    Returns:
        List of tuples: (entity_name, parent, description)
    """
    supabase_manager = get_supabase_manager(config)
    project_name = os.path.basename(working_dir)
    
    try:
        result = supabase_manager.supabase.table('entities').select(
            'entity_name, parent, description'
        ).eq(
            'entity_name', entity_name
        ).eq(
            'metadata->>project', project_name
        ).execute()
        
        entities = []
        for row in result.data:
            entities.append((row['entity_name'], row['parent'], row['description']))
        
        return entities
        
    except Exception as e:
        logger.error(f"Error searching for entity {entity_name}: {e}")
        return []


def insert_data_to_supabase(working_dir: str, config: dict):
    """
    Insert entity, relation, and community data to Supabase.
    
    Args:
        working_dir: Working directory containing data files
        config: Configuration dictionary
    """
    logger.info(f"Inserting relational data to Supabase for {working_dir}")
    
    supabase_manager = get_supabase_manager(config)
    project_name = os.path.basename(working_dir)
    
    # Get project-specific table names (to match what create_project_tables creates)
    sanitized = "".join(c if c.isalnum() else '_' for c in project_name).lower().strip('_')
    
    # Clear existing relational data for this project from project-specific tables
    tables_to_clear = [f"{sanitized}_relations", f"{sanitized}_communities", f"{sanitized}_text_units"]
    for table in tables_to_clear:
        try:
            # Use gt (greater than) with a timestamp in the past to match all rows
            result = supabase_manager.supabase.table(table).delete().gt('created_at', '1970-01-01T00:00:00Z').execute()
            logger.info(f"Cleared existing {table} data for project: {project_name}")
        except Exception as e:
            logger.warning(f"Could not clear existing {table} data: {e}")
    
    # Insert entities (non-vector data)
    entity_path = os.path.join(working_dir, "all_entities.json")
    if os.path.exists(entity_path):
        entities_to_insert = []
        
        try:
            with open(entity_path, "r") as f:
                for level, line in enumerate(f):
                    local_entity = json.loads(line)
                    
                    if isinstance(local_entity, dict):
                        entities = [local_entity]
                    else:
                        entities = local_entity
                    
                    for entity in entities:
                        # Only insert relational data here (embeddings handled separately)
                        entity_data = {
                            'entity_name': entity.get('entity_name', ''),
                            'description': entity.get('description', ''),
                            'source_id': "|".join(entity.get('source_id', '').split("|")[:5]) if entity.get('source_id') else '',
                            'degree': entity.get('degree', 0),
                            'parent': entity.get('parent', ''),
                            'level': level,
                            'metadata': {'project': project_name}
                        }
                        entities_to_insert.append(entity_data)
            
            # Insert entities in batches to project-specific table
            entities_table = f"{sanitized}_entities"
            batch_size = 100
            total_inserted = 0
            for i in range(0, len(entities_to_insert), batch_size):
                batch = entities_to_insert[i:i + batch_size]
                try:
                    result = supabase_manager.supabase.table(entities_table).insert(batch).execute()
                    batch_inserted = len(result.data) if result.data else 0
                    total_inserted += batch_inserted
                    logger.info(f"Inserted entity batch {i//batch_size + 1}: {batch_inserted} entities")
                except Exception as e:
                    logger.error(f"Error inserting entity batch: {e}")
            
            logger.info(f"✅ Inserted {total_inserted} entities")
            
        except Exception as e:
            logger.error(f"Error processing entities file: {e}")
    
    # Insert relations
    relation_path = os.path.join(working_dir, "generate_relations.json")
    if os.path.exists(relation_path):
        relations_to_insert = []
        
        try:
            with open(relation_path, "r") as f:
                for line in f:
                    relation = json.loads(line)
                    relation_data = {
                        'src_tgt': relation.get('src_tgt', ''),
                        'tgt_src': relation.get('tgt_src', ''),
                        'description': relation.get('description', ''),
                        'weight': relation.get('weight', 1),
                        'level': relation.get('level', 0),
                        'metadata': {'project': project_name}
                    }
                    relations_to_insert.append(relation_data)
            
            # Insert relations in batches to project-specific table
            relations_table = f"{sanitized}_relations"
            batch_size = 100
            total_inserted = 0
            for i in range(0, len(relations_to_insert), batch_size):
                batch = relations_to_insert[i:i + batch_size]
                try:
                    result = supabase_manager.supabase.table(relations_table).insert(batch).execute()
                    batch_inserted = len(result.data) if result.data else 0
                    total_inserted += batch_inserted
                    logger.info(f"Inserted relation batch {i//batch_size + 1}: {batch_inserted} relations")
                except Exception as e:
                    logger.error(f"Error inserting relation batch: {e}")
            
            logger.info(f"✅ Inserted {total_inserted} relations")
            
        except Exception as e:
            logger.error(f"Error processing relations file: {e}")
    
    # Insert communities
    community_path = os.path.join(working_dir, "community.json")
    if os.path.exists(community_path):
        communities_to_insert = []
        
        try:
            with open(community_path, "r") as f:
                for line in f:
                    community = json.loads(line)
                    community_data = {
                        'entity_name': community.get('entity_name', ''),
                        'entity_description': community.get('entity_description', ''),
                        'findings': str(community.get('findings', '')),
                        'metadata': {'project': project_name}
                    }
                    communities_to_insert.append(community_data)
            
            # Insert communities in batches to project-specific table
            communities_table = f"{sanitized}_communities"
            batch_size = 100
            total_inserted = 0
            for i in range(0, len(communities_to_insert), batch_size):
                batch = communities_to_insert[i:i + batch_size]
                try:
                    result = supabase_manager.supabase.table(communities_table).insert(batch).execute()
                    batch_inserted = len(result.data) if result.data else 0
                    total_inserted += batch_inserted
                    logger.info(f"Inserted community batch {i//batch_size + 1}: {batch_inserted} communities")
                except Exception as e:
                    logger.error(f"Error inserting community batch: {e}")
            
            logger.info(f"✅ Inserted {total_inserted} communities")
            
        except Exception as e:
            logger.error(f"Error processing communities file: {e}")
    
    # Insert text units (chunks) from chunk.json
    chunk_path = os.path.join(working_dir, "chunk.json")
    if os.path.exists(chunk_path):
        try:
            with open(chunk_path, "r") as f:
                chunks_data = json.load(f)
            
            logger.info(f"Processing {len(chunks_data)} text units from chunk.json")
            insert_text_units_supabase(working_dir, chunks_data, config)
            
        except Exception as e:
            logger.error(f"Error processing chunks file: {e}")
    else:
        logger.warning(f"Chunks file not found: {chunk_path}")
        logger.info("Text units will not be inserted")

def insert_text_units_supabase(working_dir: str, chunks_data: List[Dict], config: dict):
    """
    Insert text units (chunks) to Supabase.
    
    Args:
        working_dir: Working directory for project identification
        chunks_data: List of chunk dictionaries
        config: Configuration dictionary
    """
    supabase_manager = get_supabase_manager(config)
    project_name = os.path.basename(working_dir)
    
    text_units = []
    for chunk in chunks_data:
        # Use existing hash_code if available, otherwise create one
        hash_code = chunk.get('hash_code')
        text = chunk.get('text', '')
        
        if not hash_code:
            # Create hash of the text for deduplication if not provided
            hash_code = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        text_unit_data = {
            'hash_code': hash_code,
            'text': text,
            'metadata': {
                'project': project_name,
                'chunk_id': chunk.get('id', ''),
                'source': chunk.get('source', ''),
                'original_hash': chunk.get('hash_code', hash_code)  # Store original hash for reference
            }
        }
        text_units.append(text_unit_data)
    
    # Get project-specific table name for text_units
    sanitized = "".join(c if c.isalnum() else '_' for c in project_name).lower().strip('_')
    text_units_table = f"{sanitized}_text_units"
    
    # Insert in batches to project-specific table
    batch_size = 100
    total_inserted = 0
    for i in range(0, len(text_units), batch_size):
        batch = text_units[i:i + batch_size]
        try:
            # Use upsert for text_units since hash_code has a unique constraint
            result = supabase_manager.supabase.table(text_units_table).upsert(
                batch, on_conflict='hash_code'
            ).execute()
            batch_inserted = len(result.data) if result.data else 0
            total_inserted += batch_inserted
            logger.info(f"Inserted text units batch {i//batch_size + 1}: {batch_inserted} units")
        except Exception as e:
            logger.error(f"Error inserting text units batch: {e}")
            # Fallback to regular insert if upsert fails
            try:
                result = supabase_manager.supabase.table(text_units_table).insert(batch).execute()
                batch_inserted = len(result.data) if result.data else 0
                total_inserted += batch_inserted
                logger.info(f"Fallback insert for text units batch {i//batch_size + 1}: {batch_inserted} units")
            except Exception as fallback_e:
                logger.error(f"Both upsert and insert failed for text units batch: {fallback_e}")
    
    logger.info(f"✅ Inserted {total_inserted} text units")

# LeanRAG-specific search algorithms (Supabase implementations)

def find_tree_root_supabase(entity_name: str, working_dir: str, config: dict) -> List[str]:
    """
    Find tree path from entity to root using Supabase (replaces MySQL find_tree_root).
    
    This implements the core LeanRAG hierarchical traversal algorithm.
    """
    try:
        supabase_manager = get_supabase_manager(config)
        project_name = os.path.basename(working_dir)
        entities_table = get_project_table_name('entities', project_name)
        
        res = [entity_name]
        entity = entity_name
        
        # Get max level to avoid infinite loops
        max_level_result = supabase_manager.supabase.table(entities_table).select('level').order('level', desc=True).limit(1).execute()
        max_level = max_level_result.data[0]['level'] if max_level_result.data else 10
        
        i = 0
        while i < max_level:
            # Find parent entity
            parent_result = supabase_manager.supabase.table(entities_table).select('parent').eq('entity_name', entity).execute()
            
            if not parent_result.data:
                break
                
            parent = parent_result.data[0]['parent']
            if not parent or parent == entity:  # Avoid infinite loops
                break
                
            entity = parent
            res.append(entity)
            i += 1
        
        return res
        
    except Exception as e:
        logger.error(f"Error in find_tree_root_supabase: {e}")
        return [entity_name]  # Return at least the original entity

def find_path_supabase(entity1: str, entity2: str, working_dir: str, config: dict, level: int = 0, depth: int = 5) -> Optional[List[str]]:
    """
    Find path between two entities using Supabase (replaces MySQL find_path).
    
    This implements LeanRAG's graph path finding with recursive logic.
    """
    try:
        supabase_manager = get_supabase_manager(config)
        project_name = os.path.basename(working_dir)
        relations_table = get_project_table_name('relations', project_name)
        
        # Since Supabase doesn't support recursive CTEs as easily as MySQL,
        # we'll implement iterative breadth-first search
        
        visited = set()
        queue = [(entity1, [entity1])]
        
        for current_depth in range(depth):
            next_queue = []
            
            while queue:
                current_entity, path = queue.pop(0)
                
                if current_entity == entity2:
                    return path
                
                if current_entity in visited:
                    continue
                    
                visited.add(current_entity)
                
                # Find all connected entities at the specified level
                relations_result = supabase_manager.supabase.table(relations_table).select('tgt_src').eq('src_tgt', current_entity).eq('level', level).execute()
                
                for relation in relations_result.data:
                    next_entity = relation['tgt_src']
                    if next_entity not in visited and next_entity not in [p for p in path]:  # Avoid cycles
                        new_path = path + [next_entity]
                        next_queue.append((next_entity, new_path))
            
            queue = next_queue
            if not queue:
                break
        
        return None  # No path found
        
    except Exception as e:
        logger.error(f"Error in find_path_supabase: {e}")
        return None

def search_nodes_link_supabase(entity1: str, entity2: str, working_dir: str, config: dict, level: int = 0) -> Optional[tuple]:
    """
    Search for relationship between two entities using Supabase (replaces MySQL search_nodes_link).
    """
    try:
        supabase_manager = get_supabase_manager(config)
        project_name = os.path.basename(working_dir)
        relations_table = get_project_table_name('relations', project_name)
        
        # Try forward direction
        result = supabase_manager.supabase.table(relations_table).select('*').eq('src_tgt', entity1).eq('tgt_src', entity2).execute()
        
        if result.data:
            relation = result.data[0]
            return (relation['src_tgt'], relation['tgt_src'], relation['description'], relation['weight'], relation['level'])
        
        # Try reverse direction
        result = supabase_manager.supabase.table(relations_table).select('*').eq('src_tgt', entity2).eq('tgt_src', entity1).execute()
        
        if result.data:
            relation = result.data[0]
            return (relation['src_tgt'], relation['tgt_src'], relation['description'], relation['weight'], relation['level'])
        
        return None
        
    except Exception as e:
        logger.error(f"Error in search_nodes_link_supabase: {e}")
        return None

def search_nodes_supabase(entity_set: List[str], working_dir: str, config: dict) -> List[tuple]:
    """
    Search for entity information using Supabase (replaces MySQL search_nodes).
    """
    try:
        supabase_manager = get_supabase_manager(config)
        project_name = os.path.basename(working_dir)
        entities_table = get_project_table_name('entities', project_name)
        
        res = []
        for entity in entity_set:
            result = supabase_manager.supabase.table(entities_table).select('*').eq('entity_name', entity).eq('level', 0).execute()
            
            if result.data:
                entity_data = result.data[0]
                res.append((
                    entity_data['entity_name'],
                    entity_data['description'], 
                    entity_data['source_id'],
                    entity_data['degree'],
                    entity_data['parent'],
                    entity_data['level']
                ))
        
        return res
        
    except Exception as e:
        logger.error(f"Error in search_nodes_supabase: {e}")
        return []

def search_community_supabase(entity_name: str, working_dir: str, config: dict) -> Optional[tuple]:
    """
    Search for community description using Supabase (replaces MySQL search_community).
    """
    try:
        supabase_manager = get_supabase_manager(config)
        project_name = os.path.basename(working_dir)
        communities_table = get_project_table_name('communities', project_name)
        
        result = supabase_manager.supabase.table(communities_table).select('*').eq('entity_name', entity_name).execute()
        
        if result.data:
            community = result.data[0]
            return (community['entity_name'], community['entity_description'], community['findings'])
        
        return None
        
    except Exception as e:
        logger.error(f"Error in search_community_supabase: {e}")
        return None

def get_text_units_supabase(working_dir: str, hash_codes: List[str], config: dict, k: int = 5) -> str:
    """
    Get text units by hash codes using Supabase (replaces MySQL get_text_units).
    
    This implements LeanRAG's frequency-based text unit selection algorithm.
    """
    try:
        supabase_manager = get_supabase_manager(config)
        project_name = os.path.basename(working_dir)
        text_units_table = get_project_table_name('text_units', project_name)
        
        # Flatten hash codes (handle | delimited codes)
        chunks_list = []
        for chunks in hash_codes:
            if "|" in chunks:
                temp_chunks = chunks.split("|")
            else:
                temp_chunks = [chunks]
            chunks_list += temp_chunks
        
        # Count frequency (replicate Counter logic)
        from collections import Counter
        counter = Counter(chunks_list)
        
        # Select most frequent items
        duplicates = [item for item, _ in sorted(
            [(item, count) for item, count in counter.items() if count > 1],
            key=lambda x: x[1],
            reverse=True
        )[:k]]
        
        # Fill remaining slots with unique items if needed
        if len(duplicates) < k:
            used = set(duplicates)
            for item, _ in counter.items():
                if item not in used:
                    duplicates.append(item)
                    used.add(item)
                if len(duplicates) == k:
                    break
        
        # Retrieve text content from database
        text_units = ""
        if duplicates:
            result = supabase_manager.supabase.table(text_units_table).select('hash_code', 'text').in_('hash_code', duplicates).execute()
            
            # Create hash -> text mapping
            chunks_dict = {item['hash_code']: item['text'] for item in result.data}
            
            # Build final text units string
            for hash_code in duplicates:
                if hash_code in chunks_dict:
                    text_units += chunks_dict[hash_code] + "\n"
        
        return text_units
        
    except Exception as e:
        logger.error(f"Error in get_text_units_supabase: {e}")
        return ""

def search_chunks_supabase(working_dir: str, entity_set: List[str], config: dict) -> List[str]:
    """
    Search for source_id chunks for entities using Supabase.
    """
    try:
        supabase_manager = get_supabase_manager(config)
        project_name = os.path.basename(working_dir)
        entities_table = get_project_table_name('entities', project_name)
        
        res = []
        for entity in entity_set:
            if entity == 'root':
                continue
                
            result = supabase_manager.supabase.table(entities_table).select('source_id').eq('entity_name', entity).execute()
            
            if result.data:
                res.append(result.data[0]['source_id'])
        
        return res
        
    except Exception as e:
        logger.error(f"Error in search_chunks_supabase: {e}")
        return []

def get_supabase_stats(working_dir: str, config: dict) -> Dict[str, Any]:
    """
    Get comprehensive statistics about the Supabase database.
    
    Args:
        working_dir: Working directory for project identification
        config: Configuration dictionary
        
    Returns:
        Dictionary with database statistics
    """
    supabase_manager = get_supabase_manager(config)
    project_name = os.path.basename(working_dir)
    
    try:
        stats = supabase_manager.get_table_stats(project_name)
        
        # Add additional metrics
        stats['project_name'] = project_name
        stats['supabase_url'] = supabase_manager.url
        stats['connection_status'] = supabase_manager.test_connection()
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting Supabase stats: {e}")
        return {'error': str(e)}
    
def create_db_tables(config: dict):
    """Create database tables."""
    supabase_manager = get_supabase_manager(config['config'])
    project_name = os.path.basename(config['working_dir'])
    return supabase_manager.create_project_tables(project_name)
