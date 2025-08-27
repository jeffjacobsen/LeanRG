"""
LLM Manager for GraphExtraction that supports multiple API providers:
- Lambda Labs API
- OpenAI API
- Groq API
- Cerebras API
- Local models (vLLM/Ollama)
"""

import asyncio
import json
import time
import logging
from typing import List, Dict, Any, Optional
import aiohttp
from openai import AsyncOpenAI
import tiktoken
from collections import deque
import re
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Token counting
tokenizer = tiktoken.get_encoding("cl100k_base")

def truncate_text(text: str, max_tokens: int = 4096) -> str:
    """Truncate text to maximum token count."""
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.decode(tokens)

def estimate_tokens(text: str) -> int:
    """Estimate token count for text."""
    return len(tokenizer.encode(text))

# Provider-specific rate limits (tokens per minute)
PROVIDER_RATE_LIMITS = {
    "groq": 250000,    # 250K TPM for free tier
    "openai": 1000000,  # 1M TPM typical
    "lambda": 500000,   # Estimated
    "cerebras": 300000, # Estimated
}

class APILLMManager:
    """Unified LLM manager for OpenAI-compatible APIs (Lambda, OpenAI, etc.)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.api_provider = config.get("api_provider", "openai")
        self.base_url = config.get("llm_url", "https://api.openai.com/v1")
        self.api_key = config.get("llm_api_key")
        self.model = config.get("llm_model", "gpt-3.5-turbo")
        self.max_concurrent = config.get("max_concurrent", 10)
        self.timeout = config.get("timeout", 240)
        self.max_retries = config.get("max_retries", 3)
        
        # Rate limiting configuration
        self.rate_limit_tpm = config.get("rate_limit_tpm", PROVIDER_RATE_LIMITS.get(self.api_provider, 250000))
        self.enable_rate_limiting = config.get("enable_rate_limiting", True)
        
        # HTTP logging configuration
        self.verbose_http_logs = config.get("verbose_http_logs", False)
        if not self.verbose_http_logs:
            # Suppress HTTP request logs for cleaner output
            for noisy in ["httpx", "openai", "urllib3", "aiohttp", "asyncio"]:
                logging.getLogger(noisy).setLevel(logging.WARNING)

        # Initialize async OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
        
        # Semaphore for concurrent requests
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Rate limiting tracking
        self.token_usage_history = deque()  # (timestamp, tokens_used)
        self.rate_limit_lock = asyncio.Lock()
        
        # Statistics
        self.total_tokens = 0
        self.total_requests = 0
        
        logger.info(f"Initialized {self.api_provider.upper()} LLM Manager")
        logger.info(f"Model: {self.model}")
        logger.info(f"Base URL: {self.base_url}")
        logger.info(f"Max concurrent: {self.max_concurrent}")
        logger.info(f"Rate limiting: {self.enable_rate_limiting} (TPM: {self.rate_limit_tpm})")
    
    async def _clean_old_usage_records(self):
        """Remove usage records older than 1 minute."""
        current_time = time.time()
        while self.token_usage_history and current_time - self.token_usage_history[0][0] > 60:
            self.token_usage_history.popleft()
    
    async def _get_current_token_usage(self) -> int:
        """Get current token usage in the last minute."""
        await self._clean_old_usage_records()
        return sum(tokens for _, tokens in self.token_usage_history)
    
    async def _wait_for_rate_limit(self, estimated_tokens: int):
        """Wait if necessary to respect rate limits."""
        if not self.enable_rate_limiting:
            return
        
        async with self.rate_limit_lock:
            current_usage = await self._get_current_token_usage()
            
            if current_usage + estimated_tokens > self.rate_limit_tpm:
                # Calculate how long to wait
                oldest_record_time = self.token_usage_history[0][0] if self.token_usage_history else time.time()
                wait_time = 61 - (time.time() - oldest_record_time)  # Wait until oldest record expires + 1 second buffer
                
                if wait_time > 0:
                    logger.info(f"Rate limit protection: waiting {wait_time:.2f}s (current usage: {current_usage}, estimated request: {estimated_tokens})")
                    await asyncio.sleep(wait_time)
    
    async def _record_token_usage(self, tokens_used: int):
        """Record token usage for rate limiting."""
        if self.enable_rate_limiting:
            async with self.rate_limit_lock:
                self.token_usage_history.append((time.time(), tokens_used))
    
    async def _parse_rate_limit_error(self, error_msg: str) -> Optional[float]:
        """Parse rate limit error message to extract retry delay."""
        # Look for patterns like "Please try again in 2.01816s"
        match = re.search(r'try again in (\d+(?:\.\d+)?)s', error_msg)
        if match:
            return float(match.group(1))
        return None
    
    async def generate_text_async(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        history_messages: List[Dict[str, str]] = None,
        **kwargs
    ) -> str:
        """
        Generate text using async API call with rate limiting.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            history_messages: Previous conversation history
            **kwargs: Additional parameters for the API
            
        Returns:
            Generated text response
        """
        async with self.semaphore:
            return await self._make_api_call(prompt, system_prompt, history_messages, **kwargs)
    
    async def _make_api_call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: List[Dict[str, str]] = None,
        **kwargs
    ) -> str:
        """Make the actual API call with retry logic."""
        
        # Prepare messages
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add history if provided
        if history_messages:
            # Truncate history to prevent token overflow
            for msg in history_messages:
                if msg.get("content"):
                    msg["content"] = truncate_text(msg["content"], max_tokens=3000)
            messages.extend(history_messages)
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        # Estimate token usage for rate limiting
        total_content = ""
        for msg in messages:
            total_content += msg.get("content", "")
        estimated_tokens = estimate_tokens(total_content)
        
        # Apply rate limiting
        await self._wait_for_rate_limit(estimated_tokens)
        
        # Retry logic
        for attempt in range(self.max_retries):
            try:
                self.total_requests += 1
                
                # Make API call based on provider
                if self.api_provider in ["lambda", "openai", "groq", "cerebras"]:
                    response = await self._openai_compatible_call(messages, **kwargs)
                else:
                    raise ValueError(f"Unsupported API provider: {self.api_provider}")
                
                # Track tokens if available
                tokens_used = estimated_tokens  # fallback
                if hasattr(response, 'usage') and response.usage:
                    tokens_used = response.usage.total_tokens
                    self.total_tokens += tokens_used
                
                # Record token usage for rate limiting
                await self._record_token_usage(tokens_used)
                
                return response.choices[0].message.content
                
            except Exception as e:
                error_str = str(e)
                logger.warning(f"API call attempt {attempt + 1} failed: {error_str}")
                
                # Check if this is a rate limit error
                if "429" in error_str or "rate limit" in error_str.lower():
                    # Try to parse the retry delay from the error message
                    retry_delay = await self._parse_rate_limit_error(error_str)
                    if retry_delay:
                        logger.info(f"Rate limit hit, waiting {retry_delay}s as suggested by API")
                        await asyncio.sleep(retry_delay + 1)  # Add 1 second buffer
                        continue
                
                if attempt == self.max_retries - 1:
                    logger.error(f"All {self.max_retries} attempts failed for prompt: {prompt[:100]}...")
                    return ""
                
                # Exponential backoff for non-rate-limit errors
                if "429" not in error_str and "rate limit" not in error_str.lower():
                    await asyncio.sleep(2 ** attempt)
        
        return ""
    
    async def _openai_compatible_call(self, messages: List[Dict], **kwargs) -> Any:
        """Make OpenAI-compatible API call."""
        
        # Prepare request parameters
        request_params = {
            "model": self.model,
            "messages": messages,
            "timeout": self.timeout,
            **kwargs
        }
        
        # Add provider-specific parameters
        if self.api_provider == "lambda":
            # Lambda-specific settings if needed
            pass
        elif self.api_provider == "openai":
            # OpenAI-specific settings if needed
            pass
        
        # Make the call
        return await self.client.chat.completions.create(**request_params)
    
    def get_stats(self) -> Dict[str, int]:
        """Get usage statistics."""
        return {
            "total_tokens": self.total_tokens,
            "total_requests": self.total_requests,
            "api_provider": self.api_provider,
            "model": self.model
        }
    
    def reset_stats(self):
        """Reset usage statistics."""
        self.total_tokens = 0
        self.total_requests = 0


class SyncLLMWrapper:
    """Synchronous wrapper for async LLM managers."""
    
    def __init__(self, async_manager):
        self.async_manager = async_manager
        
    def generate_text(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """Synchronous wrapper for generate_text_async."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.async_manager.generate_text_async(prompt, system_prompt, **kwargs)
        )
    
    def get_stats(self):
        """Get usage statistics."""
        return self.async_manager.get_stats()
    
    def __getattr__(self, name):
        """Delegate other attributes to the async manager."""
        return getattr(self.async_manager, name)


def create_llm_manager(config: Dict[str, Any]) -> Any:
    """Factory function to create appropriate LLM manager based on config."""
    
    api_provider = config.get("api_provider", "openai").lower()
    
    if api_provider in ["lambda", "openai", "groq", "cerebras"]:
        return APILLMManager(config)
    else:
        raise ValueError(f"Unsupported API provider: {api_provider}")


def create_sync_llm_manager(config: Dict[str, Any]) -> SyncLLMWrapper:
    """Factory function to create synchronous LLM manager wrapper."""
    async_manager = create_llm_manager(config)
    return SyncLLMWrapper(async_manager)