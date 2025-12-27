"""
DeepSeek Provider for LLM Alpha.

Implements connection to DeepSeek models.
"""

import os
import sys
from typing import Awaitable, Callable, List, Optional

from llmalpha.agent.providers.base import LLMProvider, LLMResponse, Message


class DeepSeekProvider(LLMProvider):
    """
    DeepSeek provider.

    Example:
        provider = DeepSeekProvider(model="deepseek-chat")
        response = await provider.complete([
            Message(role="user", content="Hello!")
        ])
    """

    name = "deepseek"

    def __init__(
        self,
        model: str = "deepseek-reasoner",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize DeepSeek provider.

        Args:
            model: DeepSeek model identifier (deepseek-reasoner for R1, deepseek-chat)
            api_key: API key (defaults to DEEPSEEK_API_KEY env var)
            base_url: API base URL (defaults to https://api.deepseek.com)
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(model, **kwargs)

        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key required. Set DEEPSEEK_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # DeepSeek uses OpenAI-compatible API
        self.base_url = base_url or "https://api.deepseek.com"

        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )

        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    async def complete(
        self,
        messages: List[Message],
        **kwargs,
    ) -> LLMResponse:
        """
        Send messages to DeepSeek and get response.
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )

        choice = response.choices[0]

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=choice.finish_reason or "unknown",
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
        Send messages to DeepSeek with streaming output.
        Supports optional token callback for immersive UI.
        For DeepSeek-R1, also captures reasoning_content.
        """
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
        )

        collected_content = []
        collected_reasoning = []
        in_code_block = False
        code_block_announced = False
        buffer = ""
        in_reasoning = False

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta

                # Handle reasoning_content for DeepSeek-R1
                reasoning_content = getattr(delta, 'reasoning_content', None)
                if reasoning_content:
                    collected_reasoning.append(reasoning_content)
                    # Stream reasoning as thinking
                    if on_token:
                        if not in_reasoning:
                            in_reasoning = True
                        await on_token(reasoning_content)
                        continue
                    elif print_output:
                        # In legacy mode, show reasoning with special prefix
                        if not in_reasoning:
                            print("\n[Reasoning] ", end="", flush=True)
                            in_reasoning = True
                        print(reasoning_content, end="", flush=True)
                        continue

                # Handle regular content
                content = delta.content
                if content:
                    if in_reasoning:
                        in_reasoning = False
                        if print_output and not on_token:
                            print("\n[Response] ", end="", flush=True)

                    collected_content.append(content)

                    # If we have a token callback (immersive mode), use it
                    if on_token:
                        await on_token(content)
                        continue

                    # Legacy print mode (when no callback provided)
                    if print_output:
                        buffer += content

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
                            print(content, end="", flush=True)
                            buffer = ""

        if print_output and not on_token:
            print()

        full_content = "".join(collected_content)
        full_reasoning = "".join(collected_reasoning)

        # Include reasoning in response metadata if available
        response = LLMResponse(
            content=full_content,
            model=self.model,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            finish_reason="stop",
            raw_response={"reasoning_content": full_reasoning} if full_reasoning else None,
        )

        return response
