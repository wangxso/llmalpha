"""
Ollama Provider for LLM Alpha.

Implements connection to local Ollama models.
"""

from typing import List, Optional

from llmalpha.agent.providers.base import LLMProvider, LLMResponse, Message


class OllamaProvider(LLMProvider):
    """
    Ollama local model provider.

    Example:
        provider = OllamaProvider(model="codellama:34b")
        response = await provider.complete([
            Message(role="user", content="Hello!")
        ])
    """

    name = "ollama"

    def __init__(
        self,
        model: str = "codellama:34b",
        base_url: str = "http://localhost:11434",
        **kwargs,
    ):
        """
        Initialize Ollama provider.

        Args:
            model: Ollama model name
            base_url: Ollama server URL
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(model, **kwargs)
        self.base_url = base_url.rstrip("/")

        try:
            import httpx
            self._httpx = httpx
        except ImportError:
            raise ImportError(
                "httpx package required. Install with: pip install httpx"
            )

    async def complete(
        self,
        messages: List[Message],
        **kwargs,
    ) -> LLMResponse:
        """
        Send messages to Ollama and get response.

        Args:
            messages: List of chat messages
            **kwargs: Additional arguments for the API call

        Returns:
            LLMResponse
        """
        async with self._httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": m.role, "content": m.content}
                        for m in messages
                    ],
                    "options": {
                        "temperature": kwargs.get("temperature", self.temperature),
                        "num_predict": kwargs.get("max_tokens", self.max_tokens),
                    },
                    "stream": False,
                },
                timeout=120.0,  # Local models can be slow
            )

            response.raise_for_status()
            data = response.json()

        return LLMResponse(
            content=data.get("message", {}).get("content", ""),
            model=self.model,
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            },
            finish_reason="stop",
            raw_response=data,
        )

    async def list_models(self) -> List[str]:
        """
        List available models on Ollama server.

        Returns:
            List of model names
        """
        async with self._httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()

        return [model["name"] for model in data.get("models", [])]

    async def is_available(self) -> bool:
        """
        Check if Ollama server is available.

        Returns:
            True if server is reachable
        """
        try:
            async with self._httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/tags", timeout=5.0)
                return response.status_code == 200
        except Exception:
            return False
