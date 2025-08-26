"""
Qdrant Database Utilities for LeanRAG

Comprehensive Qdrant interface for vector operations including hybrid search functionality.
Consolidated from database_utils_hybrid.py for simplified management.

Key Features:
- Vector similarity search with Qdrant
- Project isolation via collection naming
- Build and search vector indices
- Integration with LeanRAG pipeline
"""

import logging
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import time
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct, Filter, FieldCondition,
        MatchValue, CollectionStatus
    )
    QDRANT_AVAILABLE = True
except ImportError:
    logger.warning("Qdrant client not available. Install with: pip install qdrant-client")
    QDRANT_AVAILABLE = False

class QdrantManager:
    """Simple Qdrant manager for vector operations."""
    
    def __init__(self, host: str = "localhost", port: int = 6333, timeout: int = 30):
        """Initialize Qdrant client."""
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant client not available")
        
        self.client = QdrantClient(host=host, port=port, timeout=timeout)
        self.host = host
        self.port = port
        logger.info(f"✅ Qdrant client initialized: {host}:{port}")
    
    def test_connection(self) -> bool:
        """Test connection to Qdrant."""
        try:
            collections = self.client.get_collections()
            logger.info("✅ Qdrant connection test passed")
            return True
        except Exception as e:
            logger.error(f"❌ Qdrant connection test failed: {e}")
            return False
    
    def create_collection(self, collection_name: str, vector_size: int) -> bool:
        """Create or ensure collection exists."""
        try:
            collection_name = f"leanrag_{collection_name}"
            
            # Check if collection exists
            try:
                collection_info = self.client.get_collection(collection_name)
                logger.info(f"📋 Collection '{collection_name}' already exists")
                return True
            except Exception:
                # Collection doesn't exist, create it
                pass
            
            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            logger.info(f"✅ Created collection '{collection_name}' with {vector_size}D vectors")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create collection '{collection_name}': {e}")
            return False
    
    def insert_vectors(self, project_name: str, entities: List[Dict], batch_size: int = 100) -> bool:
        """Insert vectors to Qdrant collection."""
        try:
            collection_name = f"leanrag_{project_name}"
            points = []
            
            for entity in entities:
                vector = entity.get('vector', [])
                if not vector:
                    continue
                
                # Convert numpy arrays to lists
                if isinstance(vector, np.ndarray):
                    vector = vector.flatten().tolist()
                
                # Create point with metadata
                point = PointStruct(
                    id=entity.get('id', len(points)),
                    vector=vector,
                    payload={
                        'entity_name': entity.get('entity_name', ''),
                        'description': entity.get('description', ''),
                        'level': entity.get('level', 0),
                        'source_id': entity.get('source_id', ''),
                        'degree': entity.get('degree', 0),
                        'parent': entity.get('parent', '')
                    }
                )
                points.append(point)
            
            if not points:
                logger.warning("No vectors to insert")
                return True
            
            # Insert in batches
            total_inserted = 0
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                result = self.client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
                total_inserted += len(batch)
                logger.info(f"Inserted batch {i//batch_size + 1}: {len(batch)} vectors")
            
            logger.info(f"✅ Successfully inserted {total_inserted} vectors to '{collection_name}' (0 failed batches)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to insert vectors: {e}")
            return False
    
    def search_vectors(self, project_name: str, query_vector: List[float], topk: int = 10, level_filter: Optional[int] = None) -> List[Dict]:
        """Search for similar vectors."""
        try:
            collection_name = f"leanrag_{project_name}"
            
            # Build filter if needed
            query_filter = None
            if level_filter is not None:
                query_filter = Filter(
                    must=[
                        FieldCondition(key="level", match=MatchValue(value=level_filter))
                    ]
                )
            
            # Search
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=topk,
                query_filter=query_filter
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'id': result.id,
                    'score': result.score,
                    'entity_name': result.payload.get('entity_name', ''),
                    'description': result.payload.get('description', ''),
                    'level': result.payload.get('level', 0),
                    'source_id': result.payload.get('source_id', ''),
                    'degree': result.payload.get('degree', 0),
                    'parent': result.payload.get('parent', '')
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Error searching vectors: {e}")
            return []

def get_qdrant_manager(config: dict) -> QdrantManager:
    """Create QdrantManager from configuration."""
    qdrant_config = config.get('qdrant_conf', {})
    host = qdrant_config.get('host', 'localhost')
    port = qdrant_config.get('port', 6333)
    timeout = qdrant_config.get('timeout', 30)
    
    return QdrantManager(host, port, timeout)

# LeanRAG-specific vector search functions

def search_vector_search_qdrant(working_dir: str, query_vector: List[float], config: dict, topk: int = 10, level_mode: int = 2) -> List[tuple]:
    """
    Search vector database using Qdrant with LeanRAG-specific level filtering.
    
    Args:
        working_dir: Working directory for project identification
        query_vector: Query vector for similarity search
        config: Configuration dictionary
        topk: Number of results to return
        level_mode: 0=leaf nodes only, 1=aggregated nodes only, 2=all nodes
        
    Returns:
        List of tuples: (entity_name, parent, description, source_id)
    """
    try:
        qdrant_manager = get_qdrant_manager(config)
        project_name = os.path.basename(working_dir)
        
        # Determine level filter based on level_mode (matching original ChromaDB logic)
        level_filter = None
        if level_mode == 0:
            level_filter = 0  # Only leaf nodes (level 0)
        elif level_mode == 1:
            # Aggregated nodes (level > 0) - we'll need to handle this with multiple searches
            # since Qdrant filtering is more limited than ChromaDB's $gt operator
            pass
        # level_mode == 2 means search all levels (no filter)
        
        # Search vectors in Qdrant
        results = qdrant_manager.search_vectors(project_name, query_vector, topk, level_filter)
        
        # Convert to format expected by query_graph.py (matching original search_vector_search)
        extract_results = []
        for result in results:
            entity_name = result.get('entity_name', '')
            parent = result.get('parent', '')
            description = result.get('description', '')
            source_id = result.get('source_id', '')
            
            extract_results.append((entity_name, parent, description, source_id))
        
        # Handle level_mode == 1 (aggregated nodes) with a second search if needed
        if level_mode == 1 and len(extract_results) < topk:
            # Search for level > 0 nodes by doing additional searches
            # This is a simplified approach - in a production system you might want
            # to modify the Qdrant search to handle level ranges better
            additional_results = []
            for level in range(1, 10):  # Search levels 1-9
                level_results = qdrant_manager.search_vectors(project_name, query_vector, topk, level)
                for result in level_results:
                    entity_name = result.get('entity_name', '')
                    parent = result.get('parent', '')
                    description = result.get('description', '')
                    source_id = result.get('source_id', '')
                    additional_results.append((entity_name, parent, description, source_id))
                
                if len(additional_results) >= topk:
                    break
            
            extract_results = additional_results[:topk]
        
        return extract_results
        
    except Exception as e:
        logger.error(f"❌ Error in Qdrant vector search: {e}")
        return []

# ========================================
# Hybrid/Build Functions (from database_utils_hybrid.py)
# ========================================

def build_vector_search_hybrid(data: List[Any], working_dir: str, config: dict) -> bool:
    """
    Build vector search using Qdrant only (no Supabase integration).
    Consolidated from database_utils_hybrid.py.
    
    Args:
        data: Entity data with embeddings (list of levels)
        working_dir: Working directory for project identification
        config: Configuration dictionary
        
    Returns:
        True if successful
    """
    if not QDRANT_AVAILABLE:
        logger.error("❌ Qdrant not available for vector search")
        return False
    
    logger.info(f"Building Qdrant vector search index for {working_dir}")
    
    try:
        qdrant_manager = get_qdrant_manager(config)
        
        # Test connection
        if not qdrant_manager.test_connection():
            raise ConnectionError("Cannot connect to Qdrant")
        
        project_name = os.path.basename(working_dir)
        
        # Flatten data structure (same logic as other implementations)
        id_counter = 0
        flatten = []
        
        for level, sublist in enumerate(data):
            if not isinstance(sublist, list):
                # Single item at this level
                item = sublist.copy()
                item['id'] = id_counter
                id_counter += 1
                item['level'] = level
                
                # Handle vector consistency
                vector = item.get('vector', [])
                if isinstance(vector, np.ndarray):
                    vector = vector.flatten().tolist()
                elif not isinstance(vector, list):
                    logger.warning(f"Invalid vector type for entity {item.get('entity_name', 'unknown')}: {type(vector)}")
                    continue
                
                item['vector_dim'] = len(vector)
                item['vector'] = vector
                flatten.append(item)
            else:
                # Multiple items at this level
                for item in sublist:
                    if not isinstance(item, dict):
                        logger.warning(f"Skipping invalid item at level {level}: {type(item)}")
                        continue
                        
                    item_copy = item.copy()
                    item_copy['id'] = id_counter
                    id_counter += 1
                    item_copy['level'] = level
                    
                    vector = item_copy.get('vector', [])
                    if isinstance(vector, np.ndarray):
                        vector = vector.flatten().tolist()
                    elif not isinstance(vector, list):
                        logger.warning(f"Invalid vector type for entity {item_copy.get('entity_name', 'unknown')}: {type(vector)}")
                        continue
                    
                    item_copy['vector_dim'] = len(vector)
                    item_copy['vector'] = vector
                    flatten.append(item_copy)
        
        if not flatten:
            logger.error("No valid entities found to insert")
            return False
        
        # Group by vector dimension
        dimension_groups = {}
        for item in flatten:
            dim = item['vector_dim']
            if dim not in dimension_groups:
                dimension_groups[dim] = []
            dimension_groups[dim].append(item)
        
        logger.info(f"Found {len(dimension_groups)} different embedding dimensions: {list(dimension_groups.keys())}")
        
        # Use most common dimension for main insertion
        main_dim = max(dimension_groups.keys(), key=lambda x: len(dimension_groups[x]))
        main_items = dimension_groups[main_dim]
        
        logger.info(f"Using dimension {main_dim} for main collection ({len(main_items)} items)")
        
        # Create collection and insert vectors to Qdrant
        if not qdrant_manager.create_collection(project_name, main_dim):
            logger.error(f"Failed to create Qdrant collection for {project_name}")
            return False
        
        # Insert vectors only (no metadata to Supabase)
        success = qdrant_manager.insert_vectors(project_name, main_items, 100)
        
        if success:
            logger.info(f"✅ Successfully built Qdrant vector index for '{project_name}'")
            
            # Log skipped dimensions
            for dim, items in dimension_groups.items():
                if dim != main_dim:
                    logger.warning(f"⚠️ Skipped {len(items)} items with dimension {dim} (not main dimension {main_dim})")
            
            return True
        else:
            logger.error(f"❌ Failed to insert vectors to Qdrant for '{project_name}'")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error building Qdrant vector index: {e}")
        return False

def build_vector_search(data, working_dir: str, config: dict):
    """Build vector search index using Qdrant only."""
    return build_vector_search_hybrid(data, working_dir, config)

def search_vector_search_hybrid(working_dir: str, query, config: dict, topk: int = 10, level_mode: int = 2):
    """
    Search vector similarity using Qdrant only.
    Consolidated from database_utils_hybrid.py.
    
    Args:
        working_dir: Working directory for project identification
        query: Query vector or text
        config: Configuration dictionary
        topk: Number of results to return
        level_mode: 0=leaf nodes only, 1=aggregated only, 2=all levels
        
    Returns:
        List of similar entities with metadata
    """
    if not QDRANT_AVAILABLE:
        logger.error("❌ Qdrant not available for vector search")
        return []
    
    try:
        qdrant_manager = get_qdrant_manager(config)
        project_name = os.path.basename(working_dir)
        
        # Determine level filter based on mode
        level_filter = None
        if level_mode == 0:
            level_filter = 0
        elif level_mode == 1:
            # For aggregated nodes, we'll handle this with multiple searches
            # since Qdrant doesn't support "greater than" filters easily
            pass
        
        # Search vectors in Qdrant
        results = qdrant_manager.search_vectors(
            project_name, query, topk, level_filter
        )
        
        # Results already contain all needed metadata from Qdrant
        return results
        
    except Exception as e:
        logger.error(f"❌ Error in Qdrant vector search: {e}")
        return []

def search_vector_search(working_dir: str, query, config: dict, topk: int = 10, level_mode: int = 2):
    """Search vector similarity using Qdrant only."""
    return search_vector_search_hybrid(working_dir, query, config, topk, level_mode)