"""
Anthropic Provider for LLM Alpha.

Implements connection to Anthropic Claude models.
"""

import os
from typing import Awaitable, Callable, List, Optional

from llmalpha.agent.providers.base import LLMProvider, LLMResponse, Message


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude provider.

    Example:
        provider = AnthropicProvider(model="claude-3-opus-20240229")
        response = await provider.complete([
            Message(role="user", content="Hello!")
        ])
    """

    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-3-opus-20240229",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Anthropic provider.

        Args:
            model: Anthropic model identifier
            api_key: API key (defaults to ANTHROPIC_API_KEY env var)
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(model, **kwargs)

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )

        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)

    async def complete(
        self,
        messages: List[Message],
        **kwargs,
    ) -> LLMResponse:
        """
        Send messages to Anthropic and get response.

        Args:
            messages: List of chat messages
            **kwargs: Additional arguments for the API call

        Returns:
            LLMResponse
        """
        # Anthropic separates system message from conversation
        system = ""
        conversation = []

        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                conversation.append({"role": m.role, "content": m.content})

        # Ensure conversation starts with user message
        if not conversation or conversation[0]["role"] != "user":
            conversation.insert(0, {"role": "user", "content": "Please help me with the following task."})

        response = await self.client.messages.create(
            model=self.model,
            system=system,
            messages=conversation,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
        )

        # Extract text content
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        return LLMResponse(
            content=content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            finish_reason=response.stop_reason or "unknown",
            raw_response=response,
        )

    async def complete_stream(
        self,
        messages: List[Message],
        print_output: bool = True,
        on_token: Optional[Callable[[str], Awaitable[None]]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send messages to Anthropic with streaming output.
        Supports optional token callback for immersive UI.
        """
        # Anthropic separates system message from conversation
        system = ""
        conversation = []

        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                conversation.append({"role": m.role, "content": m.content})

        # Ensure conversation starts with user message
        if not conversation or conversation[0]["role"] != "user":
            conversation.insert(0, {"role": "user", "content": "Please help me with the following task."})

        collected_content = []
        in_code_block = False
        code_block_announced = False
        buffer = ""

        async with self.client.messages.stream(
            model=self.model,
            system=system,
            messages=conversation,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
        ) as stream:
            async for text in stream.text_stream:
                collected_content.append(text)

                # If we have a token callback (immersive mode), use it
                if on_token:
                    await on_token(text)
                    continue

                # Legacy print mode (when no callback provided)
                if print_output:
                    buffer += text

                    # Check for code block start
                    if not in_code_block and "```" in buffer:
                        idx = buffer.find("```")
                        if idx > 0:
                            print(buffer[:idx], end="", flush=True)
                        in_code_block = True
                        if not code_block_announced:
                            print("\n\n  正在编写代码...", end="", flush=True)
                            code_block_announced = True
                        buffer = buffer[idx+3:]
                        continue

                    # Check for code block end
                    if in_code_block and "```" in buffer:
                        idx = buffer.find("```")
                        buffer = buffer[idx+3:]
                        in_code_block = False
                        continue

                    # Print if not in code block
                    if not in_code_block:
                        print(text, end="", flush=True)
                        buffer = ""

        if print_output and not on_token:
            print()  # Final newline

        full_content = "".join(collected_content)

        return LLMResponse(
            content=full_content,
            model=self.model,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            finish_reason="stop",
            raw_response=None,
        )
