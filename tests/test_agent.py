"""
Tests for the Agent module.
"""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from llmalpha.agent.config import AgentConfig, LLMConfig, ValidationConfig, IterationState
from llmalpha.agent.sandbox.executor import CodeValidator, SafeExecutor, ExecutionResult
from llmalpha.agent.providers.base import Message, LLMResponse


class TestAgentConfig:
    """Test agent configuration classes."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AgentConfig()

        assert config.max_iterations == 20
        assert config.max_consecutive_failures == 5
        assert config.early_stop_sharpe == 1.5
        assert config.sandbox_timeout == 120
        assert config.generation_type == "strategy"

    def test_llm_config(self):
        """Test LLM configuration."""
        config = LLMConfig(
            provider="anthropic",
            model="claude-3-opus-20240229",
            temperature=0.5,
        )

        assert config.provider == "anthropic"
        assert config.model == "claude-3-opus-20240229"
        assert config.temperature == 0.5

    def test_validation_config(self):
        """Test validation configuration."""
        config = ValidationConfig(
            mode="wf_only",
            min_sharpe=0.5,
            min_trades=100,
        )

        assert config.mode == "wf_only"
        assert config.min_sharpe == 0.5
        assert config.min_trades == 100


class TestIterationState:
    """Test iteration state tracking."""

    def test_initial_state(self):
        """Test initial state values."""
        state = IterationState()

        assert state.iteration == 0
        assert state.best_sharpe == 0.0
        assert state.best_hypothesis_code is None
        assert state.consecutive_failures == 0
        assert state.consecutive_successes == 0
        assert state.history == []

    def test_record_iteration_success(self):
        """Test recording a successful iteration."""
        state = IterationState()
        config = AgentConfig()

        state.record_iteration("H001", passed=True, sharpe=1.2)

        assert state.iteration == 1
        assert state.best_sharpe == 1.2
        assert state.best_hypothesis_code == "H001"
        assert state.consecutive_failures == 0
        assert state.consecutive_successes == 1

    def test_record_iteration_failure(self):
        """Test recording a failed iteration."""
        state = IterationState()

        state.record_iteration("H001", passed=False, sharpe=0.1)

        assert state.iteration == 1
        assert state.best_sharpe == 0.0  # Not updated for failure
        assert state.consecutive_failures == 1
        assert state.consecutive_successes == 0

    def test_should_stop_max_iterations(self):
        """Test stopping at max iterations."""
        state = IterationState()
        config = AgentConfig(max_iterations=3)

        state.iteration = 3
        should_stop, reason = state.should_stop(config)

        assert should_stop is True
        assert "maximum iterations" in reason

    def test_should_stop_consecutive_failures(self):
        """Test stopping on consecutive failures."""
        state = IterationState()
        config = AgentConfig(max_consecutive_failures=3)

        state.consecutive_failures = 3
        should_stop, reason = state.should_stop(config)

        assert should_stop is True
        assert "consecutive failures" in reason

    def test_should_stop_target_sharpe(self):
        """Test stopping on target Sharpe achieved."""
        state = IterationState()
        config = AgentConfig(early_stop_sharpe=1.5)

        state.best_sharpe = 1.6
        should_stop, reason = state.should_stop(config)

        assert should_stop is True
        assert "Sharpe" in reason

    def test_reset(self):
        """Test state reset."""
        state = IterationState()
        state.iteration = 5
        state.best_sharpe = 1.2
        state.best_hypothesis_code = "H005"

        state.reset()

        assert state.iteration == 0
        assert state.best_sharpe == 0.0
        assert state.best_hypothesis_code is None


class TestCodeValidator:
    """Test code validation."""

    def test_valid_code(self):
        """Test validation of safe code."""
        validator = CodeValidator()

        code = """
import numpy as np
import pandas as pd

def calculate(df):
    return df['close'].rolling(20).mean()
"""
        is_valid, error = validator.validate(code)
        assert is_valid is True
        assert error is None

    def test_forbidden_import_os(self):
        """Test rejection of os import."""
        validator = CodeValidator()

        code = "import os\nos.system('rm -rf /')"
        is_valid, error = validator.validate(code)

        assert is_valid is False
        assert "os" in error

    def test_forbidden_import_subprocess(self):
        """Test rejection of subprocess import."""
        validator = CodeValidator()

        code = "import subprocess\nsubprocess.run(['ls'])"
        is_valid, error = validator.validate(code)

        assert is_valid is False
        assert "subprocess" in error

    def test_forbidden_eval(self):
        """Test rejection of eval."""
        validator = CodeValidator()

        code = "result = eval('1 + 1')"
        is_valid, error = validator.validate(code)

        assert is_valid is False
        assert "eval" in error

    def test_forbidden_exec(self):
        """Test rejection of exec."""
        validator = CodeValidator()

        code = "exec('print(1)')"
        is_valid, error = validator.validate(code)

        assert is_valid is False
        assert "exec" in error

    def test_forbidden_open(self):
        """Test rejection of open."""
        validator = CodeValidator()

        code = "f = open('/etc/passwd', 'r')"
        is_valid, error = validator.validate(code)

        assert is_valid is False
        assert "open" in error


class TestMessage:
    """Test message dataclass."""

    def test_message_creation(self):
        """Test message creation."""
        msg = Message(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_fields(self):
        """Test message field access."""
        msg = Message(role="assistant", content="Hi there")

        assert msg.role == "assistant"
        assert msg.content == "Hi there"


class TestLLMResponse:
    """Test LLM response dataclass."""

    def test_response_creation(self):
        """Test response creation."""
        response = LLMResponse(
            content="Generated code",
            model="gpt-4",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            finish_reason="stop",
        )

        assert response.content == "Generated code"
        assert response.model == "gpt-4"
        assert response.usage["prompt_tokens"] == 100
        assert response.finish_reason == "stop"


class TestSafeExecutor:
    """Test safe code execution."""

    def test_execute_simple_code(self):
        """Test execution of simple code."""
        executor = SafeExecutor(timeout=5)

        code = "result = 1 + 1"
        result = executor.execute(code)

        assert result.success is True
        assert result.result.get("result") == 2

    def test_execute_with_numpy(self):
        """Test execution with numpy."""
        executor = SafeExecutor(timeout=5)

        code = """
import numpy as np
arr = np.array([1, 2, 3])
mean = arr.mean()
"""
        result = executor.execute(code)

        assert result.success is True
        assert result.result.get("mean") == 2.0

    def test_execute_with_pandas(self):
        """Test execution with pandas."""
        executor = SafeExecutor(timeout=5)

        code = """
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3]})
total = df['a'].sum()
"""
        result = executor.execute(code)

        assert result.success is True
        assert result.result.get("total") == 6

    def test_execute_forbidden_code(self):
        """Test rejection of forbidden code."""
        executor = SafeExecutor(timeout=5)

        code = "import os"
        result = executor.execute(code)

        assert result.success is False
        assert "forbidden" in result.error.lower() or "os" in result.error.lower()

    def test_execute_syntax_error(self):
        """Test handling of syntax errors."""
        executor = SafeExecutor(timeout=5)

        code = "def broken("
        result = executor.execute(code)

        assert result.success is False
        assert result.error is not None


class TestQwenProvider:
    """Test Qwen provider integration."""

    def test_qwen_config(self):
        """Test Qwen provider can be configured in LLMConfig."""
        config = LLMConfig(
            provider="qwen",
            model="qwen-turbo",
            temperature=0.7,
        )

        assert config.provider == "qwen"
        assert config.model == "qwen-turbo"
        assert config.temperature == 0.7

    def test_get_provider_qwen(self):
        """Test get_provider function can create QwenProvider."""
        from llmalpha.agent.providers.base import get_provider

        # Mock environment variable to avoid requiring real API key
        with patch.dict("os.environ", {"QWEN_API_KEY": "test-api-key"}):
            provider = get_provider(
                provider_name="qwen",
                model="qwen-turbo",
                api_key="test-api-key",
            )

            assert provider.name == "qwen"
            assert provider.model == "qwen-turbo"
            assert provider.api_key == "test-api-key"
            assert provider.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def test_get_provider_qwen_custom_base_url(self):
        """Test QwenProvider with custom base_url."""
        from llmalpha.agent.providers.base import get_provider

        custom_url = "https://custom-qwen-endpoint.com/v1"
        with patch.dict("os.environ", {"QWEN_API_KEY": "test-api-key"}):
            provider = get_provider(
                provider_name="qwen",
                model="qwen-plus",
                api_key="test-api-key",
                base_url=custom_url,
            )

            assert provider.name == "qwen"
            assert provider.model == "qwen-plus"
            assert provider.base_url == custom_url

    def test_qwen_provider_missing_api_key(self):
        """Test QwenProvider raises error when API key is missing."""
        from llmalpha.agent.providers.qwen import QwenProvider

        # Clear environment variable
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Qwen API key required"):
                QwenProvider(model="qwen-turbo")

    @pytest.mark.asyncio
    async def test_qwen_provider_complete(self):
        """Test QwenProvider complete method with mocked client."""
        from llmalpha.agent.providers.qwen import QwenProvider

        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello from Qwen"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "qwen-turbo"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        # Mock client
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.dict("os.environ", {"QWEN_API_KEY": "test-api-key"}):
            provider = QwenProvider(model="qwen-turbo", api_key="test-api-key")
            provider.client = mock_client

            messages = [Message(role="user", content="Hello")]
            response = await provider.complete(messages)

            assert response.content == "Hello from Qwen"
            assert response.model == "qwen-turbo"
            assert response.usage["prompt_tokens"] == 10
            assert response.usage["completion_tokens"] == 5
            assert response.usage["total_tokens"] == 15
            assert response.finish_reason == "stop"

            # Verify client was called correctly
            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs["model"] == "qwen-turbo"
            assert call_args.kwargs["messages"] == [{"role": "user", "content": "Hello"}]

    @pytest.mark.asyncio
    async def test_qwen_provider_complete_stream(self):
        """Test QwenProvider complete_stream method with mocked client."""
        from llmalpha.agent.providers.qwen import QwenProvider

        # Mock streaming response chunks
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Hello"
        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = " from"
        mock_chunk3 = MagicMock()
        mock_chunk3.choices = [MagicMock()]
        mock_chunk3.choices[0].delta.content = " Qwen"

        async def mock_stream():
            for chunk in [mock_chunk1, mock_chunk2, mock_chunk3]:
                yield chunk

        # Mock client
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        with patch.dict("os.environ", {"QWEN_API_KEY": "test-api-key"}):
            provider = QwenProvider(model="qwen-turbo", api_key="test-api-key")
            provider.client = mock_client

            messages = [Message(role="user", content="Hello")]
            response = await provider.complete_stream(messages, print_output=False)

            assert response.content == "Hello from Qwen"
            assert response.model == "qwen-turbo"
            assert response.finish_reason == "stop"

            # Verify client was called with stream=True
            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs["stream"] is True
