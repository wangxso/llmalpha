"""
Safe Code Execution Sandbox for LLM Alpha.

Provides secure execution of LLM-generated code with:
- AST-based code validation
- Forbidden module/function detection
- Timeout-protected execution
- Restricted namespace
"""

import ast
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Type

import numpy as np
import pandas as pd


@dataclass
class ExecutionResult:
    """Result of code execution."""

    success: bool
    result: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    execution_time: float = 0.0


class CodeValidator:
    """
    Validates code safety using AST analysis.

    Detects dangerous imports, function calls, and operations.
    """

    # Modules that could compromise system security
    FORBIDDEN_MODULES: Set[str] = {
        # System access
        "os", "sys", "subprocess", "shutil", "pathlib",
        # Network
        "socket", "urllib", "requests", "httpx", "aiohttp",
        # Serialization (code injection risk)
        "pickle", "marshal", "shelve",
        # Builtins manipulation
        "__builtins__", "builtins",
        # File operations
        "io", "tempfile",
        # Code execution
        "importlib", "runpy",
        # Other dangerous modules
        "ctypes", "multiprocessing", "threading",
    }

    # Dangerous function calls
    FORBIDDEN_CALLS: Set[str] = {
        "eval", "exec", "compile",
        "open", "input",
        "__import__",
        "globals", "locals", "vars",
        "getattr", "setattr", "delattr", "hasattr",
        "exit", "quit",
        "breakpoint",
    }

    # Dangerous attribute accesses
    FORBIDDEN_ATTRIBUTES: Set[str] = {
        "__class__", "__bases__", "__subclasses__",
        "__code__", "__globals__", "__closure__",
        "__builtins__",
    }

    def validate(self, code: str) -> tuple[bool, Optional[str]]:
        """
        Validate code safety.

        Args:
            code: Python code string

        Returns:
            (is_safe, error_message)
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module in self.FORBIDDEN_MODULES:
                        return False, f"Forbidden module: {alias.name}"

            # Check from imports
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split(".")[0]
                    if module in self.FORBIDDEN_MODULES:
                        return False, f"Forbidden module: {node.module}"

            # Check function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.FORBIDDEN_CALLS:
                        return False, f"Forbidden function: {node.func.id}"

            # Check attribute access
            if isinstance(node, ast.Attribute):
                if node.attr in self.FORBIDDEN_ATTRIBUTES:
                    return False, f"Forbidden attribute: {node.attr}"

        return True, None

    def count_lines(self, code: str) -> int:
        """Count non-empty, non-comment lines."""
        lines = code.split("\n")
        count = 0
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                count += 1
        return count


class SafeExecutor:
    """
    Executes code in a restricted sandbox.

    Example:
        executor = SafeExecutor(timeout=60)
        result = executor.execute(code)

        if result.success:
            strategy_class = result.result.get("MyStrategy")
    """

    def __init__(
        self,
        timeout: int = 60,
        max_lines: int = 200,
    ):
        """
        Initialize executor.

        Args:
            timeout: Maximum execution time in seconds
            max_lines: Maximum allowed code lines
        """
        self.timeout = timeout
        self.max_lines = max_lines
        self.validator = CodeValidator()

    def get_safe_namespace(self) -> Dict[str, Any]:
        """
        Create a restricted namespace for code execution.

        Returns:
            Namespace dictionary with safe imports
        """
        # Import LLM Alpha base classes
        from llmalpha.factors.base import Factor, FactorMeta
        from llmalpha.signals.base import Signal, SignalResult
        from llmalpha.strategies.base import Strategy, StrategyConfig

        return {
            # Data science libraries
            "pd": pd,
            "np": np,
            "pandas": pd,
            "numpy": np,

            # LLM Alpha base classes
            "Factor": Factor,
            "FactorMeta": FactorMeta,
            "Signal": Signal,
            "SignalResult": SignalResult,
            "Strategy": Strategy,
            "StrategyConfig": StrategyConfig,

            # Safe builtins
            "range": range,
            "len": len,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
            "sorted": sorted,
            "reversed": reversed,
            "zip": zip,
            "map": map,
            "filter": filter,
            "enumerate": enumerate,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "type": type,
            "callable": callable,
            "any": any,
            "all": all,

            # Type constructors
            "dict": dict,
            "list": list,
            "tuple": tuple,
            "set": set,
            "frozenset": frozenset,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "bytes": bytes,

            # Constants
            "True": True,
            "False": False,
            "None": None,

            # Exceptions (for error handling in generated code)
            "Exception": Exception,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "ZeroDivisionError": ZeroDivisionError,
        }

    def execute(
        self,
        code: str,
        additional_namespace: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """
        Safely execute code.

        Args:
            code: Python code to execute
            additional_namespace: Extra variables for namespace
            timeout: Override default timeout (for LLM-estimated values)

        Returns:
            ExecutionResult
        """
        # Use provided timeout or fall back to default
        effective_timeout = timeout if timeout is not None else self.timeout

        # Validate code safety
        is_safe, error = self.validator.validate(code)
        if not is_safe:
            return ExecutionResult(
                success=False,
                error=error,
                error_type="SecurityError",
            )

        # Check code length
        line_count = self.validator.count_lines(code)
        if line_count > self.max_lines:
            return ExecutionResult(
                success=False,
                error=f"Code too long: {line_count} lines (max {self.max_lines})",
                error_type="ValidationError",
            )

        # Prepare namespace
        namespace = self.get_safe_namespace()
        if additional_namespace:
            namespace.update(additional_namespace)

        # Execute with timeout
        start_time = time.time()

        def run_code():
            exec(code, namespace)
            return namespace

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_code)
                result_namespace = future.result(timeout=effective_timeout)

            execution_time = time.time() - start_time

            return ExecutionResult(
                success=True,
                result=result_namespace,
                execution_time=execution_time,
            )

        except FutureTimeoutError:
            return ExecutionResult(
                success=False,
                error=f"Execution timed out after {effective_timeout} seconds",
                error_type="TimeoutError",
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                error_type=type(e).__name__,
            )

    def extract_class(
        self,
        namespace: Dict[str, Any],
        base_class: Type,
    ) -> Optional[Type]:
        """
        Find a class that inherits from base_class in namespace.

        Args:
            namespace: Execution namespace
            base_class: Expected base class

        Returns:
            Found class or None
        """
        for name, obj in namespace.items():
            if (
                isinstance(obj, type)
                and issubclass(obj, base_class)
                and obj is not base_class
            ):
                return obj
        return None

    def execute_and_extract(
        self,
        code: str,
        base_class: Type,
        additional_namespace: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> tuple[ExecutionResult, Optional[Type]]:
        """
        Execute code and extract the generated class.

        Args:
            code: Python code to execute
            base_class: Expected base class for the generated class
            additional_namespace: Extra variables for namespace
            timeout: Override default timeout (for LLM-estimated values)

        Returns:
            (ExecutionResult, extracted_class or None)
        """
        result = self.execute(code, additional_namespace, timeout=timeout)

        if not result.success:
            return result, None

        extracted = self.extract_class(result.result, base_class)

        if not extracted:
            result.success = False
            result.error = f"No valid {base_class.__name__} subclass found"
            result.error_type = "ExtractionError"
            return result, None

        return result, extracted
