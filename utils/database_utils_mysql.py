#!/usr/bin/env python3
"""
MySQL Database Utilities for LeanRAG

Clean MySQL interface that integrates with the config system.
Provides MySQL-based implementations of LeanRAG search algorithms.
"""

import json
import os
import pymysql
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import Counter
import logging

# Setup logging
logger = logging.getLogger(__name__)

class MySQLManager:
    """MySQL database manager for LeanRAG operations."""
    
    def __init__(self, host: str, port: int, user: str, password: str, 
                 charset: str = 'utf8mb4', autocommit: bool = True, 
                 connect_timeout: int = 10):
        """Initialize MySQL connection."""
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.charset = charset
        self.autocommit = autocommit
        self.connect_timeout = connect_timeout
        self._connection = None
        logger.info(f"✅ MySQL manager initialized: {host}:{port}")
    
    def get_connection(self, db_name: str = None):
        """Get MySQL connection, creating if needed."""
        try:
            if self._connection is None or not self._connection.open:
                self._connection = pymysql.connect(
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    passwd=self.password,
                    charset=self.charset,
                    database=db_name,
                    autocommit=self.autocommit,
                    connect_timeout=self.connect_timeout
                )
            return self._connection
        except Exception as e:
            logger.error(f"❌ Failed to connect to MySQL: {e}")
            raise
    
    def close_connection(self):
        """Close MySQL connection."""
        if self._connection and self._connection.open:
            self._connection.close()
            self._connection = None
    
    def test_connection(self) -> bool:
        """Test MySQL connection."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            logger.info("✅ MySQL connection test passed")
            return True
        except Exception as e:
            logger.error(f"❌ MySQL connection test failed: {e}")
            return False

# Global MySQL manager cache
_mysql_manager_cache = None

def get_mysql_manager(config: dict) -> MySQLManager:
    """Create MySQLManager from configuration with caching."""
    global _mysql_manager_cache
    
    mysql_config = config.get('mysql_conf', {})
    host = mysql_config.get('host', 'localhost')
    port = mysql_config.get('port', 4321)
    user = mysql_config.get('user', 'root')
    password = mysql_config.get('password', '123')
    charset = mysql_config.get('charset', 'utf8mb4')
    autocommit = mysql_config.get('autocommit', True)
    connect_timeout = mysql_config.get('connect_timeout', 10)
    
    # Create cache key based on connection parameters
    cache_key = f"{host}:{port}:{user}:{password[:5]}"
    
    # Return cached manager if exists and matches current config
    if _mysql_manager_cache is not None:
        cached_key = f"{_mysql_manager_cache.host}:{_mysql_manager_cache.port}:{_mysql_manager_cache.user}:{_mysql_manager_cache.password[:5]}"
        if cached_key == cache_key:
            return _mysql_manager_cache
    
    # Create new manager and cache it
    _mysql_manager_cache = MySQLManager(host, port, user, password, charset, autocommit, connect_timeout)
    return _mysql_manager_cache

# LeanRAG-specific search functions using original database_utils.py logic

def create_db_table_mysql_config(working_dir: str, config: dict):
    """Create MySQL database tables using config."""
    mysql_manager = get_mysql_manager(config)
    db_name = os.path.basename(working_dir)
    
    try:
        # Create database if not exists
        conn = mysql_manager.get_connection()
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
        cursor.close()
        
        # Connect to specific database
        conn = mysql_manager.get_connection(db_name)
        cursor = conn.cursor()
        
        # Create entities table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS `{db_name}`.entities (
                entity_name VARCHAR(255) PRIMARY KEY,
                description TEXT,
                source_id TEXT,
                degree INT DEFAULT 0,
                parent VARCHAR(255) DEFAULT '',
                level INT DEFAULT 0
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        
        # Create relations table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS `{db_name}`.relations (
                src_tgt VARCHAR(255),
                tgt_src VARCHAR(255),
                description TEXT,
                weight INT DEFAULT 1,
                level INT DEFAULT 0,
                PRIMARY KEY (src_tgt, tgt_src)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        
        # Create community table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS `{db_name}`.community (
                entity_name VARCHAR(255) PRIMARY KEY,
                entity_description TEXT,
                findings TEXT
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        
        # Create text_units table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS `{db_name}`.text_units (
                hash_code VARCHAR(255) PRIMARY KEY,
                text TEXT
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        
        # Create indexes for performance
        try:
            cursor.execute(f"CREATE INDEX idx_entities_level ON `{db_name}`.entities (level)")
            cursor.execute(f"CREATE INDEX idx_entities_parent ON `{db_name}`.entities (parent)")
            cursor.execute(f"CREATE INDEX idx_relations_src ON `{db_name}`.relations (src_tgt)")
            cursor.execute(f"CREATE INDEX idx_relations_tgt ON `{db_name}`.relations (tgt_src)")
        except pymysql.Error:
            # Indexes might already exist, ignore errors
            pass
        
        cursor.close()
        logger.info(f"✅ MySQL tables created for database: {db_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating MySQL tables: {e}")
        return False

def insert_data_to_mysql_config(working_dir: str, config: dict):
    """Insert data to MySQL using config."""
    mysql_manager = get_mysql_manager(config)
    db_name = os.path.basename(working_dir)
    
    try:
        conn = mysql_manager.get_connection(db_name)
        cursor = conn.cursor()
        
        # Insert entities
        entity_path = os.path.join(working_dir, "all_entities.json")
        if os.path.exists(entity_path):
            with open(entity_path, 'r') as f:
                entities = []
                for line in f:
                    line = line.strip()
                    if line:
                        entities.extend(json.loads(line))  # Each line is a JSON array
            
            for entity in entities:
                # Handle both dict objects and potential string objects
                if isinstance(entity, dict):
                    cursor.execute(f"""
                        INSERT IGNORE INTO `{db_name}`.entities 
                        (entity_name, description, source_id, degree, parent, level) 
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        entity.get('entity_name', ''),
                        entity.get('description', ''),
                        entity.get('source_id', ''),
                        entity.get('degree', 0),
                        entity.get('parent', ''),
                        entity.get('level', 0)
                    ))
                else:
                    logger.warning(f"Skipping invalid entity entry: {type(entity)}")
            logger.info(f"✅ Inserted {len(entities)} entities")
        
        # Insert relations
        relation_path = os.path.join(working_dir, "generate_relations.json")
        if os.path.exists(relation_path):
            with open(relation_path, 'r') as f:
                relations = []
                for line in f:
                    line = line.strip()
                    if line:
                        relations.append(json.loads(line))
            
            for relation in relations:
                # Handle both dict objects and potential string objects
                if isinstance(relation, dict):
                    cursor.execute(f"""
                        INSERT IGNORE INTO `{db_name}`.relations 
                        (src_tgt, tgt_src, description, weight, level) 
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        relation.get('src_tgt', ''),
                        relation.get('tgt_src', ''),
                        relation.get('description', ''),
                        relation.get('weight', 1),
                        relation.get('level', 0)
                    ))
                else:
                    logger.warning(f"Skipping invalid relation entry: {type(relation)}")
            logger.info(f"✅ Inserted {len(relations)} relations")
        
        # Insert communities
        community_path = os.path.join(working_dir, "community.json")
        if os.path.exists(community_path):
            with open(community_path, 'r') as f:
                communities = []
                for line in f:
                    line = line.strip()
                    if line:
                        communities.append(json.loads(line))
            
            for community in communities:
                # Handle both dict objects and potential string objects
                if isinstance(community, dict):
                    findings = community.get('findings', '')
                    # Convert findings list to string if needed
                    if isinstance(findings, list):
                        findings = json.dumps(findings)
                    
                    cursor.execute(f"""
                        INSERT IGNORE INTO `{db_name}`.community 
                        (entity_name, entity_description, findings) 
                        VALUES (%s, %s, %s)
                    """, (
                        community.get('entity_name', ''),
                        community.get('entity_description', ''),
                        findings
                    ))
                else:
                    logger.warning(f"Skipping invalid community entry: {type(community)}")
            logger.info(f"✅ Inserted {len(communities)} communities")
        
        # Insert text units (chunk file)
        chunk_path = os.path.join(working_dir, "chunk.json")
        if os.path.exists(chunk_path):
            with open(chunk_path, 'r') as f:
                chunks = json.load(f)
            
            for chunk in chunks:
                cursor.execute(f"""
                    INSERT IGNORE INTO `{db_name}`.text_units 
                    (hash_code, text) 
                    VALUES (%s, %s)
                """, (
                    chunk.get('hash_code', ''),
                    chunk.get('text', '')
                ))
            logger.info(f"✅ Inserted {len(chunks)} text units")
        
        cursor.close()
        logger.info("✅ MySQL data insertion completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error inserting data to MySQL: {e}")
        return False

# LeanRAG search functions - adapted from original database_utils.py

def find_tree_root_mysql(entity_name: str, working_dir: str, config: dict) -> List[str]:
    """Find tree path from entity to root using MySQL."""
    mysql_manager = get_mysql_manager(config)
    db_name = os.path.basename(working_dir)
    
    try:
        conn = mysql_manager.get_connection(db_name)
        cursor = conn.cursor()
        
        path = []
        current_entity = entity_name
        max_iterations = 100  # Prevent infinite loops
        iteration_count = 0
        
        while current_entity and current_entity != '' and iteration_count < max_iterations:
            path.append(current_entity)
            
            cursor.execute(f"SELECT parent FROM `{db_name}`.entities WHERE entity_name = %s", (current_entity,))
            result = cursor.fetchone()
            
            if result and result[0]:
                current_entity = result[0]
            else:
                break
            
            iteration_count += 1
        
        cursor.close()
        return path
        
    except Exception as e:
        logger.error(f"Error in find_tree_root_mysql: {e}")
        return [entity_name]

def search_nodes_link_mysql(entity1: str, entity2: str, working_dir: str, config: dict) -> Optional[Tuple[str, str, str]]:
    """Search for relationship between two entities using MySQL."""
    mysql_manager = get_mysql_manager(config)
    db_name = os.path.basename(working_dir)
    
    try:
        conn = mysql_manager.get_connection(db_name)
        cursor = conn.cursor()
        
        # Try direct match first
        cursor.execute(f"SELECT * FROM `{db_name}`.relations WHERE src_tgt = %s AND tgt_src = %s", (entity1, entity2))
        result = cursor.fetchone()
        
        if not result:
            # Try reverse match
            cursor.execute(f"SELECT * FROM `{db_name}`.relations WHERE src_tgt = %s AND tgt_src = %s", (entity2, entity1))
            result = cursor.fetchone()
        
        cursor.close()
        
        if result:
            return (result[0], result[1], result[2])  # src_tgt, tgt_src, description
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error in search_nodes_link_mysql: {e}")
        return None

def search_community_mysql(entity_name: str, working_dir: str, config: dict) -> Optional[Tuple[str, str, str]]:
    """Search for community description using MySQL."""
    mysql_manager = get_mysql_manager(config)
    db_name = os.path.basename(working_dir)
    
    try:
        conn = mysql_manager.get_connection(db_name)
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT * FROM `{db_name}`.community WHERE entity_name = %s", (entity_name,))
        result = cursor.fetchone()
        
        cursor.close()
        
        if result:
            return (result[0], result[1], result[2])  # entity_name, entity_description, findings
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error in search_community_mysql: {e}")
        return None

def get_text_units_mysql(working_dir: str, hash_codes: List[str], config: dict, k: int = 5) -> str:
    """Get text units by hash codes using MySQL."""
    mysql_manager = get_mysql_manager(config)
    db_name = os.path.basename(working_dir)
    
    logger.info(f"🔍 get_text_units_mysql called with {len(hash_codes) if hash_codes else 0} hash_codes: {hash_codes[:3] if hash_codes else 'None'}")
    
    if not hash_codes:
        logger.warning("No hash codes provided to get_text_units_mysql")
        return ""
    
    try:
        conn = mysql_manager.get_connection(db_name)
        cursor = conn.cursor()
        
        # First check what hash codes exist in database
        cursor.execute(f"SELECT COUNT(*) FROM `{db_name}`.text_units")
        total_count = cursor.fetchone()[0]
        logger.info(f"📊 Total text_units in database: {total_count}")
        
        # Get text units for the hash codes
        placeholders = ','.join(['%s'] * len(hash_codes))
        cursor.execute(f"SELECT hash_code, text FROM `{db_name}`.text_units WHERE hash_code IN ({placeholders})", hash_codes)
        results = cursor.fetchall()
        
        logger.info(f"📋 Found {len(results)} matching text units for {len(hash_codes)} hash codes")
        
        cursor.close()
        
        if results:
            texts = [result[1] for result in results]  # result[1] is text, result[0] is hash_code
            return '\n'.join(texts[:k])
        else:
            logger.warning(f"No text units found for hash codes: {hash_codes[:5]}")
            return ""
            
    except Exception as e:
        logger.error(f"Error in get_text_units_mysql: {e}")
        return ""

def search_nodes_mysql(entity_name: str, working_dir: str, config: dict) -> List[Tuple[str, str, str]]:
    """Search for entity information using MySQL."""
    mysql_manager = get_mysql_manager(config)
    db_name = os.path.basename(working_dir)
    
    try:
        conn = mysql_manager.get_connection(db_name)
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT entity_name, parent, description FROM `{db_name}`.entities WHERE entity_name = %s", (entity_name,))
        results = cursor.fetchall()
        
        cursor.close()
        return results
        
    except Exception as e:
        logger.error(f"Error in search_nodes_mysql: {e}")
        return []