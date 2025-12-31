"""
LLM Provider Base Classes for LLM Alpha.

Defines abstract interfaces for LLM providers.
"""

import asyncio
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional


@dataclass
class Message:
    """Chat message structure."""

    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """LLM response structure."""

    content: str
    model: str
    usage: Dict[str, int]  # Token usage
    finish_reason: str
    raw_response: Any = None


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Subclasses implement connections to OpenAI, Anthropic, Ollama, etc.
    """

    name: str = "base"

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        """
        Initialize the provider.

        Args:
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    async def complete(
        self,
        messages: List[Message],
        **kwargs,
    ) -> LLMResponse:
        """
        Send messages and get a response.

        Args:
            messages: List of chat messages
            **kwargs: Provider-specific arguments

        Returns:
            LLMResponse
        """
        pass

    async def complete_stream(
        self,
        messages: List[Message],
        print_output: bool = True,
        on_token: Optional[Callable[[str], Awaitable[None]]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send messages with streaming output (default falls back to complete).

        Args:
            messages: List of chat messages
            print_output: Whether to print tokens as they arrive
            on_token: Optional async callback for each token (for UI updates)
            **kwargs: Provider-specific arguments

        Returns:
            LLMResponse
        """
        # Default implementation: fall back to non-streaming
        response = await self.complete(messages, **kwargs)
        if print_output:
            print(response.content)
        if on_token:
            # For non-streaming fallback, send entire content as single token
            await on_token(response.content)
        return response

    async def complete_with_retry(
        self,
        messages: List[Message],
        max_retries: int = 3,
        base_delay: float = 1.0,
        **kwargs,
    ) -> LLMResponse:
        """
        Complete with exponential backoff retry.

        Args:
            messages: List of chat messages
            max_retries: Maximum retry attempts
            base_delay: Base delay between retries
            **kwargs: Provider-specific arguments

        Returns:
            LLMResponse

        Raises:
            Exception: If all retries fail
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                return await self.complete(messages, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)

        raise last_error

    def extract_code(self, response: str) -> Optional[str]:
        """
        Extract Python code from LLM response.

        Args:
            response: LLM response text

        Returns:
            Extracted Python code or None
        """
        # Try to find ```python ... ``` blocks
        pattern = r"```python\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Try to find ``` ... ``` blocks
        pattern = r"```\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        return None

    def extract_all_code_blocks(self, response: str) -> List[str]:
        """
        Extract all code blocks from response.

        Args:
            response: LLM response text

        Returns:
            List of code blocks
        """
        pattern = r"```(?:python)?\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        return [m.strip() for m in matches]


def get_provider(
    provider_name: str,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
) -> LLMProvider:
    """
    Factory function to create LLM provider.

    Args:
        provider_name: Provider name ("openai", "anthropic", "ollama", "deepseek", "qwen")
        model: Model identifier
        api_key: API key (if required)
        base_url: Base URL (for Ollama or custom endpoints)
        **kwargs: Additional provider-specific arguments

    Returns:
        LLMProvider instance

    Raises:
        ValueError: If provider is not supported
    """
    if provider_name == "openai":
        from llmalpha.agent.providers.openai import OpenAIProvider
        return OpenAIProvider(model=model, api_key=api_key, **kwargs)

    elif provider_name == "anthropic":
        from llmalpha.agent.providers.anthropic import AnthropicProvider
        return AnthropicProvider(model=model, api_key=api_key, **kwargs)

    elif provider_name == "deepseek":
        from llmalpha.agent.providers.deepseek import DeepSeekProvider
        return DeepSeekProvider(model=model, api_key=api_key, base_url=base_url, **kwargs)

    elif provider_name == "qwen":
        from llmalpha.agent.providers.qwen import QwenProvider
        return QwenProvider(model=model, api_key=api_key, base_url=base_url, **kwargs)

    elif provider_name == "ollama":
        from llmalpha.agent.providers.ollama import OllamaProvider
        return OllamaProvider(
            model=model,
            base_url=base_url or "http://localhost:11434",
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown provider: {provider_name}")
