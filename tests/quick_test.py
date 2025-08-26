#!/usr/bin/env python3
"""
Quick test script for GraphExtraction API configuration.
Tests Lambda Labs, OpenAI, and local LLM configurations.
"""

import argparse
import sys
import os
import yaml
import time
import asyncio
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.llm_manager import create_llm_manager
import logging
from openai import AsyncOpenAI
import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_llm_api(llm_manager, test_prompt="Hello, this is a test."):
    """Test LLM API with a simple prompt."""
    try:
        logger.info(f"Testing LLM with prompt: '{test_prompt}'")
        start_time = time.time()
        
        response = await llm_manager.generate_text_async(test_prompt)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if response and len(response.strip()) > 0:
            logger.info(f"✅ LLM API test successful!")
            logger.info(f"   Response time: {response_time:.2f}s")
            logger.info(f"   Response length: {len(response)} characters")
            logger.info(f"   Response preview: {response[:100]}{'...' if len(response) > 100 else ''}")
            return True
        else:
            logger.error("❌ LLM API test failed: Empty response")
            return False
            
    except Exception as e:
        logger.error(f"❌ LLM API test failed: {str(e)}")
        return False

async def test_model_listing(config):
    """Test model listing functionality."""
    try:
        logger.info("Testing model listing...")
        
        api_provider = config.get("api_provider", "openai").lower()
        base_url = config.get("llm_url", "https://api.openai.com/v1")
        api_key = config.get("llm_api_key")
        
        if api_provider in ["lambda", "openai", "groq", "cerebras"]:
            # Use OpenAI-compatible endpoint
            client = AsyncOpenAI(base_url=base_url, api_key=api_key)
            models = await client.models.list()
            
            logger.info(f"✅ Found {len(models.data)} available models:")
            for model in models.data[:5]:  # Show first 5 models
                logger.info(f"   - {model.id}")
            if len(models.data) > 5:
                logger.info(f"   ... and {len(models.data) - 5} more")
            return True
        else:
            logger.info("Model listing not supported for this provider")
            return True
            
    except Exception as e:
        logger.error(f"❌ Model listing failed: {str(e)}")
        return False

def load_config(config_path: str):
    """Load configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config {config_path}: {e}")
        return None

async def main():
    parser = argparse.ArgumentParser(
        description="Quick test script for GraphExtraction API configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/quick_test.py                                           # Test default config
  python tests/quick_test.py -c config_lambda_example.yaml           # Test Lambda API example
  python tests/quick_test.py -c config_openai_example.yaml           # Test OpenAI API example
  python tests/quick_test.py --list-models                            # List available models
  python tests/quick_test.py --list-configs                           # List config files
        """
    )
    
    parser.add_argument('-c', '--config', 
                       default='config.yaml',
                       help='Configuration file path (default: config.yaml)')
    parser.add_argument('--list-models', 
                       action='store_true',
                       help='List available models from API provider')
    parser.add_argument('--list-configs', 
                       action='store_true',
                       help='List available configuration files')
    parser.add_argument('--skip-api', 
                       action='store_true',
                       help='Skip API tests, only validate config loading')
    
    args = parser.parse_args()
    
    # List available configs if requested
    if args.list_configs:
        # Look for example config files in current directory
        config_files = list(Path(".").glob("*example*.yaml"))
        if config_files:
            logger.info("Available configuration files:")
            for config_file in sorted(config_files):
                logger.info(f"  {config_file}")
        else:
            logger.error("No example configuration files found in current directory")
        return
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        logger.error(f"Failed to load configuration from {args.config}")
        return
    
    # Extract LLM configuration
    llm_config = config.get('llm_conf', {})
    if not llm_config:
        logger.error("No 'llm_conf' section found in configuration")
        return
    
    logger.info("="*60)
    logger.info("GraphExtraction API Configuration Test")
    logger.info("="*60)
    logger.info(f"Config file: {args.config}")
    logger.info(f"API Provider: {llm_config.get('api_provider', 'unknown')}")
    logger.info(f"Model: {llm_config.get('llm_model', 'unknown')}")
    logger.info(f"Base URL: {llm_config.get('llm_url', 'unknown')}")
    
    if args.skip_api:
        logger.info("✅ Configuration loaded successfully. Skipping API tests.")
        return
    
    # Create LLM manager
    try:
        llm_manager = create_llm_manager(llm_config)
        logger.info("✅ LLM manager created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create LLM manager: {e}")
        return
    
    # List available models if requested
    if args.list_models:
        success = await test_model_listing(llm_config)
        if not success:
            return
    
    # Test LLM API
    logger.info("-" * 40)
    success = await test_llm_api(llm_manager, "What is machine learning?")
    
    if success:
        logger.info("="*60)
        logger.info("✅ All tests passed! Configuration is working correctly.")
        logger.info("="*60)
    else:
        logger.error("="*60)
        logger.error("❌ Tests failed! Please check your configuration.")
        logger.error("="*60)

if __name__ == "__main__":
    asyncio.run(main())