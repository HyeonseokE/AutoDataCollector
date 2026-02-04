"""
LLM Utilities for Multi-Robot Code Generation

This module re-exports LLM utilities from the existing code_gen_lerobot module.
No duplication - just re-use.
"""

# Re-export all LLM utilities from the original module
from code_gen_lerobot.llm import (
    llm_response,
    detect_provider,
    _call_vllm_server,
    _call_cloud_server,
)
from code_gen_lerobot.llm_utils.openai_utils import chatgpt_response

try:
    from code_gen_lerobot.llm_utils.gemini import gemini_response
except ImportError:
    gemini_response = None

try:
    from code_gen_lerobot.llm_utils.llama import llama_response
except ImportError:
    llama_response = None

__all__ = [
    "llm_response",
    "detect_provider",
    "chatgpt_response",
    "gemini_response",
    "llama_response",
    "_call_vllm_server",
    "_call_cloud_server",
]
