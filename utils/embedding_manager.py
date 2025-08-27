"""
Embedding Manager for LeanRAG

Supports multiple embedding providers:
- FastEmbed: Local embedding generation with various models
- OpenAI: API-based embeddings (fallback/comparison)

FastEmbed models available:
- BAAI/bge-small-en-v1.5 (384 dim) - Fast, lightweight
- BAAI/bge-base-en-v1.5 (768 dim) - Balanced performance/quality  
- BAAI/bge-large-en-v1.5 (1024 dim) - High quality
- sentence-transformers/all-MiniLM-L6-v2 (384 dim) - Very fast
"""

import logging
import time
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import asyncio
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    logger.warning("FastEmbed not available. Install with: pip install fastembed")
    FASTEMBED_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI not available. Install with: pip install openai")
    OPENAI_AVAILABLE = False

# FastEmbed model configurations
FASTEMBED_MODELS = {
    "bge-small": {
        "name": "BAAI/bge-small-en-v1.5",
        "dimensions": 384,
        "description": "Fast, lightweight model for quick processing"
    },
    "bge-base": {
        "name": "BAAI/bge-base-en-v1.5", 
        "dimensions": 768,
        "description": "Balanced performance and quality"
    },
    "bge-large": {
        "name": "BAAI/bge-large-en-v1.5",
        "dimensions": 1024,
        "description": "High quality embeddings"
    },
    "all-minilm": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimensions": 384,
        "description": "Very fast processing, good quality"
    }
}

class EmbeddingManager:
    """Unified embedding manager supporting FastEmbed and OpenAI."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize embedding manager with configuration."""
        self.config = config
        self.embedding_config = config.get('embedding_conf', {})
        self.provider = self.embedding_config.get('provider', 'fastembed')
        
        # Initialize based on provider
        self.embedding_model = None
        self.openai_client = None
        self.dimensions = 0
        
        if self.provider == 'fastembed':
            self._init_fastembed()
        elif self.provider == 'openai':
            self._init_openai()
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")
            
        logger.info(f"✅ Embedding manager initialized: {self.provider} ({self.dimensions} dims)")
    
    def _init_fastembed(self):
        """Initialize FastEmbed model."""
        if not FASTEMBED_AVAILABLE:
            raise ImportError("FastEmbed not available. Install with: pip install fastembed")
        
        model_key = self.embedding_config.get('model', 'bge-base')
        
        if model_key in FASTEMBED_MODELS:
            model_config = FASTEMBED_MODELS[model_key]
            model_name = model_config['name']
            self.dimensions = model_config['dimensions']
        else:
            # Direct model name provided
            model_name = model_key
            # Try to infer dimensions (will be set after first embedding)
            self.dimensions = self.embedding_config.get('dimensions', 768)
        
        try:
            # Initialize FastEmbed with specified device
            device = self.embedding_config.get('device', 'cpu')
            self.embedding_model = TextEmbedding(
                model_name=model_name,
                max_length=512,  # Reasonable context length
                device=device
            )
            
            logger.info(f"✅ FastEmbed model loaded: {model_name} on {device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load FastEmbed model {model_name}: {e}")
            raise
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI client not available. Install with: pip install openai")
        
        api_key = self.embedding_config.get('api_key')
        base_url = self.embedding_config.get('base_url', 'https://api.openai.com/v1')
        
        if not api_key:
            raise ValueError("OpenAI API key not provided in embedding_conf")
        
        self.openai_client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # Set dimensions based on model
        model = self.embedding_config.get('model', 'text-embedding-3-small')
        if 'large' in model:
            self.dimensions = 3072
        elif '3-small' in model or 'ada-002' in model:
            self.dimensions = 1536
        else:
            self.dimensions = 1536  # Default
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            
        Returns:
            numpy array of embeddings (num_texts, dimensions)
        """
        if not texts:
            return np.array([])
        
        if self.provider == 'fastembed':
            return self._embed_fastembed(texts, batch_size)
        elif self.provider == 'openai':
            return self._embed_openai(texts, batch_size)
    
    def _embed_fastembed(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Generate embeddings using FastEmbed."""
        try:
            logger.info(f"🔄 Generating FastEmbed embeddings for {len(texts)} texts...")
            start_time = time.time()
            
            # FastEmbed handles batching internally
            embeddings_generator = self.embedding_model.embed(texts)
            embeddings_list = list(embeddings_generator)
            
            # Convert to numpy array
            embeddings = np.array(embeddings_list)
            
            # Update dimensions if not set
            if embeddings.shape[1] != self.dimensions and self.dimensions == 0:
                self.dimensions = embeddings.shape[1]
                logger.info(f"📏 Updated dimensions to {self.dimensions}")
            
            duration = time.time() - start_time
            logger.info(f"✅ Generated {len(embeddings)} FastEmbed embeddings in {duration:.2f}s")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ FastEmbed embedding failed: {e}")
            raise
    
    def _embed_openai(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        try:
            logger.info(f"🔄 Generating OpenAI embeddings for {len(texts)} texts...")
            start_time = time.time()
            
            model = self.embedding_config.get('model', 'text-embedding-3-small')
            embeddings_list = []
            
            # Process in batches to respect API limits
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                response = self.openai_client.embeddings.create(
                    input=batch_texts,
                    model=model
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                embeddings_list.extend(batch_embeddings)
                
                # Rate limiting
                if i + batch_size < len(texts):
                    time.sleep(0.1)  # Small delay between batches
            
            embeddings = np.array(embeddings_list)
            
            duration = time.time() - start_time
            logger.info(f"✅ Generated {len(embeddings)} OpenAI embeddings in {duration:.2f}s")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ OpenAI embedding failed: {e}")
            raise
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        embeddings = self.embed_texts([text])
        return embeddings[0] if len(embeddings) > 0 else np.array([])
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model."""
        info = {
            'provider': self.provider,
            'dimensions': self.dimensions
        }
        
        if self.provider == 'fastembed':
            model_key = self.embedding_config.get('model', 'bge-base')
            if model_key in FASTEMBED_MODELS:
                info.update(FASTEMBED_MODELS[model_key])
            else:
                info['name'] = model_key
        elif self.provider == 'openai':
            info['name'] = self.embedding_config.get('model', 'text-embedding-3-small')
        
        return info

def benchmark_embedding_models(texts: List[str], config_base: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Benchmark multiple embedding models on the same text data.
    
    Args:
        texts: Sample texts for benchmarking
        config_base: Base configuration to modify for different models
        
    Returns:
        Dictionary with model results and performance metrics
    """
    results = {}
    
    # FastEmbed models to benchmark
    fastembed_models = ['bge-small', 'bge-base', 'bge-large', 'all-minilm']
    
    for model_key in fastembed_models:
        if not FASTEMBED_AVAILABLE:
            logger.warning(f"Skipping {model_key}: FastEmbed not available")
            continue
        
        try:
            # Create config for this model
            test_config = config_base.copy()
            test_config['embedding_conf'] = {
                'provider': 'fastembed',
                'model': model_key,
                'device': 'cpu'
            }
            
            # Initialize and test
            manager = EmbeddingManager(test_config)
            
            start_time = time.time()
            embeddings = manager.embed_texts(texts[:10])  # Use first 10 texts for speed
            duration = time.time() - start_time
            
            results[model_key] = {
                'model_info': manager.get_model_info(),
                'embeddings_shape': embeddings.shape,
                'processing_time': duration,
                'texts_per_second': len(texts[:10]) / duration if duration > 0 else 0,
                'success': True
            }
            
            logger.info(f"✅ Benchmarked {model_key}: {embeddings.shape} in {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Failed to benchmark {model_key}: {e}")
            results[model_key] = {
                'success': False,
                'error': str(e)
            }
    
    return results

def get_embedding_manager(config: Dict[str, Any]) -> EmbeddingManager:
    """Factory function to create embedding manager from config."""
    return EmbeddingManager(config)

# Compatibility functions for existing codebase
def create_embedding_function(config: Dict[str, Any]):
    """Create embedding function compatible with existing code."""
    manager = EmbeddingManager(config)
    
    def embedding_function(texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """Embedding function that handles both single texts and lists."""
        if isinstance(texts, str):
            return manager.embed_single(texts)
        else:
            return manager.embed_texts(texts)
    
    return embedding_function

class PickleableEmbeddingFunction:
    """Pickleable embedding function for multiprocessing (build_graph.py style)."""
    
    def __init__(self, config: Dict[str, Any]):
        # Store config for recreation in worker processes
        self.config = config
        self.provider = config.get('provider', 'openai')
        # Don't store the client - recreate in __call__
        
    def __call__(self, texts) -> np.ndarray:
        """Generate embeddings (creates client per call for multiprocessing safety)."""
        if self.provider == 'fastembed':
            return self._generate_fastembed_embeddings(texts)
        else:
            return self._generate_openai_embeddings(texts)
    
    def _generate_fastembed_embeddings(self, texts) -> np.ndarray:
        """Generate embeddings using FastEmbed."""
        try:
            from fastembed import TextEmbedding
            import numpy as np
            
            # Handle single string input
            if isinstance(texts, str):
                texts = [texts]
            
            # Map model names
            model_name = self.config.get('model', 'bge-base')
            model_mapping = {
                'bge-small': 'BAAI/bge-small-en-v1.5',
                'bge-base': 'BAAI/bge-base-en-v1.5', 
                'bge-large': 'BAAI/bge-large-en-v1.5',
                'all-minilm': 'sentence-transformers/all-MiniLM-L6-v2'
            }
            
            full_model_name = model_mapping.get(model_name, model_name)
            device = self.config.get('device', 'cpu')
            
            # Create FastEmbed model (will be recreated in each worker)
            embedding_model = TextEmbedding(model_name=full_model_name, providers=["CPUExecutionProvider"] if device == "cpu" else None)
            
            # Generate embeddings
            embeddings = list(embedding_model.embed(texts))
            return np.array(embeddings)
            
        except Exception as e:
            raise Exception(f"FastEmbed embedding generation failed: {e}")
    
    def _generate_openai_embeddings(self, texts) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        from openai import OpenAI
        import time
        
        # Recreate client in worker process
        client = OpenAI(
            api_key=self.config['api_key'],
            base_url=self.config['base_url'],
            timeout=self.config.get('timeout', 240)
        )
        
        model = self.config['model']
        provider = self.config.get('api_provider', 'local')
        max_retries = self.config.get('max_retries', 3)
        
        # Provider-specific batch size
        batch_size = EmbeddingManager.PROVIDER_BATCH_SIZES.get(provider, 64)
        
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
        
        for attempt in range(max_retries):
            try:
                all_embeddings = []
                
                # Process in batches
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    
                    embedding_response = client.embeddings.create(
                        input=batch_texts,
                        model=model,
                    )
                    
                    batch_embeddings = [d.embedding for d in embedding_response.data]
                    all_embeddings.extend(batch_embeddings)
                
                return np.array(all_embeddings)
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        
        raise Exception("Failed to generate embeddings after all retries")



def create_pickleable_embedding_function(config: Dict[str, Any]) -> PickleableEmbeddingFunction:
    """Factory function to create pickleable embedding function for multiprocessing."""
    return PickleableEmbeddingFunction(config)