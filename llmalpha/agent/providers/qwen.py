"""
Qwen Provider for LLM Alpha.

Implements connection to Qwen models via OpenAI-compatible API.
"""

import os
from typing import Awaitable, Callable, List, Optional

from llmalpha.agent.providers.base import LLMProvider, LLMResponse, Message


class QwenProvider(LLMProvider):
    """
    Qwen provider.

    Example:
        provider = QwenProvider(model="qwen-turbo")
        response = await provider.complete([
            Message(role="user", content="Hello!")
        ])
    """

    name = "qwen"

    def __init__(
        self,
        model: str = "qwen-turbo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Qwen provider.

        Args:
            model: Qwen model identifier (qwen-turbo, qwen-plus, qwen-max, etc.)
            api_key: API key (defaults to QWEN_API_KEY env var)
            base_url: API base URL (defaults to https://dashscope.aliyuncs.com/compatible-mode/v1)
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(model, **kwargs)

        self.api_key = api_key or os.environ.get("QWEN_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Qwen API key required. Set QWEN_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Qwen uses OpenAI-compatible API via DashScope
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"

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
        Send messages to Qwen and get response.
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
        Send messages to Qwen with streaming output.
        Supports optional token callback for immersive UI.
        """
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
        )

        collected_content = []
        in_code_block = False
        code_block_announced = False
        buffer = ""

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
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
                        # Print everything before the code block
                        idx = buffer.find("```")
                        if idx > 0:
                            print(buffer[:idx], end="", flush=True)
                        in_code_block = True
                        if not code_block_announced:
                            print("\n\n  正在编写代码...", end="", flush=True)
                            code_block_announced = True
                        buffer = buffer[idx+3:]  # Skip ```
                        continue

                    # Check for code block end (when already in code block)
                    if in_code_block and "```" in buffer:
                        idx = buffer.find("```")
                        buffer = buffer[idx+3:]  # Skip closing ```
                        in_code_block = False
                        continue

                    # Print if not in code block
                    if not in_code_block:
                        print(content, end="", flush=True)
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

