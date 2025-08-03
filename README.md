# Sandboxed Python

[![PyPI](https://img.shields.io/pypi/v/sandboxed-python)](https://pypi.org/project/sandboxed-python/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://github.com/fixed-ai/sandboxed-python/blob/main/LICENSE)
[![CI](https://github.com/fixed-ai/sandboxed-python/actions/workflows/ci.yml/badge.svg)](https://github.com/fixed-ai/sandboxed-python/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/fixed-ai/sandboxed-python/graph/badge.svg?token=WPZRFD0MSR)](https://codecov.io/gh/fixed-ai/sandboxed-python)


Sandboxed Python is a lightweight, secure interpreter for a restricted subset of Python (I call it "Finite Python" as it does not allow loops/recursions). It provides a **true, interop-friendly sandbox environment** that merely depends on the Python standard library and is distributed as a single file (you might include [src/sandboxed_python/finite_python.py](https://github.com/fixed-ai/sandboxed-python/blob/main/src/sandboxed_python/finite_python.py) in your project). This makes it ideal for scenarios requiring controlled execution, such as LLM (Large Language Model) tool calls, serverless functions, or plugin systems.

### Key Features
- **Limited but Secure and Usable Python Subset**: Supports essential operations like assignments, conditionals, try-except, basic data structures (lists, dicts, sets, tuples), and safe built-ins/methods. Loops are explicitly forbidden in code (e.g., no `for` or `while`), but list/tuple creation allows implicit repetition via multiplication (e.g., `[0] * 5`).
- **Easy-to-Customize Interfaces**: Abstract base class for sandboxes allows seamless extension for custom behaviors.
- **Extremely Friendly, LLM-Readable Error Reporting**: Errors include source locations, related code snippets with highlights, suggestions, and structured data for easy parsing by LLMs or agents.
    ```
    Exec error at line 4, column 0
        2 print([*{1, 2}, 3, *(4, 5), *{5: 1}])
        3 
    ---> 4 "1" - "2"
        5     
    [Original Python Backtrace]
    TypeError: unsupported operand type(s) for -: 'str' and 'str'
    ```
- **Ready-to-Use Default Sandbox**: Includes built-in protections like recursion limits, allowed-lists for functions/methods, and monitoring hooks.
- **No Explicit Loops**: Ensures finite execution, making it safe for untrusted code while still allowing practical computations.

## Installation

(Prerequisites: Python 3.12+)

### Download the Single File

Sandboxed Python is a single-file library. Download `src/sandboxed_python/finite_python.py` and place it in your project.

### Install via pip

In order to receive the latest updates (**maybe security fixes**), you can install via pip:

```
pip install sandboxed-python
```

## Usage

### Using the Default Sandbox

The default sandbox provides a secure environment with common safe built-ins (e.g., `len`, `str`, `print`) and methods (e.g., `list.append`, `str.upper`). Use `execute_fpy` to run FPy code.

```python
from sandboxed_python import execute_fpy

# Simple print statement
execute_fpy("""
print("Hello, World!")
""")

# Complex expression with unpacking (starred expressions)
try:
    execute_fpy("""
print([*{1, 2}, 3, *(4, 5), *{5: 1}])

"1" - "2"
    """)
except FPyException as e:
    e.pretty_print()
```

Output:
```
[FPy Output] Hello, World!
[FPy Output] [1, 2, 3, 4, 5, 5]
Exec error at line 4, column 0
     2 print([*{1, 2}, 3, *(4, 5), *{5: 1}])
     3 
---> 4 "1" - "2"
     5     
[Original Python Backtrace]
TypeError: unsupported operand type(s) for -: 'str' and 'str'
```

The default sandbox outputs results prefixed with `[FPy Output]` or `[FPy Result]` for expression statements, ensuring safe and observable execution.

### Using the Custom Sandbox

Sandboxed Python is designed with seamless integration into agent systems (e.g., LLM agents like those in LangChain or AutoGPT) in mind. You can subclass `PySandbox` to customize variable storage, function/method allowances, error handling, and more. This allows tailoring the sandbox to specific agent needs, such as integrating with external APIs or state management.

```python
from sandboxed_python import (
    PySandbox, SourceLocation, execute_fpy,
    Undef, undef,
    FPyVal, # you can also use `typing.Any`...
)
import typing

class MySandbox(PySandbox):
    def __init__(self):
        self.variables: dict[str, FPyVal] = {}
        self.location: SourceLocation | None = None

    def get_location(self) -> SourceLocation | None:
        return self.location

    def set_location(self, location: SourceLocation) -> None:
        self.location = location

    def func_call(
        self, func_name: str, args: list[FPyVal], location: SourceLocation
    ) -> FPyVal:
        # Custom implementation: e.g., only allow 'len'
        if func_name == "len" and len(args) == 1:
            return len(args[0])
        if func_name == "print":
            print(*args)
            return
        raise Exception(f"Function {func_name} not allowed")

    def method_call(
        self,
        subject: FPyVal,
        method_name: str,
        args: list[FPyVal],
        location: SourceLocation,
    ) -> FPyVal:
        # Custom implementation: e.g., allow str.upper
        if isinstance(subject, str) and method_name == "upper":
            return subject.upper()
        raise Exception(f"Method {method_name} not allowed on {type(subject)}")

    def get_var(self, name: str) -> FPyVal | Undef:
        return self.variables.get(name, Undef.undef)

    def set_var(self, name: str, value: FPyVal | Undef) -> None:
        if isinstance(value, Undef):
            self.variables.pop(name, None)
        else:
            self.variables[name] = value  # type: ignore

    def list_vars(self) -> typing.Iterable[str]:
        return self.variables.keys()

    def display(self, value: FPyVal) -> None:
        print(f"[Custom Output] {value!r}")

    def exception_to_message(self, exception: Exception) -> str:
        return str(exception)

# Usage
sandbox = MySandbox()
execute_fpy("print(len('hello'))", sandbox=sandbox)
```

This customization enables agents to define domain-specific functions (e.g., API calls) while maintaining security.

### Using Sandboxed Python to Replace Traditional Tool Calls

My point here: **Python DSL is the best tool call mechanism for ALL LLMs.**

```python
from sandboxed_python import execute_fpy, DefaultSecureSandbox

class AgentTool:
    def __init__(self):
        self.sandbox = DefaultSecureSandbox()
        self.sandbox.add_function("query_db", self._query_db)  # Custom tool

    def _query_db(self, query: str) -> list[dict]:
        # Simulate safe DB query
        return [{"result": "data"}]

    def call_tool(self, generated_code: str):
        try:
            execute_fpy(generated_code, sandbox=self.sandbox)
            # Access sandbox variables for results
            return self.sandbox.get_var("result")
        except Exception as e:
            return {"error": str(e)}  # LLM can parse and retry

# Agent usage
agent = AgentTool()
llm_generated_code = """
result = query_db("SELECT * FROM users")
print(result)
"""
result = agent.call_tool(llm_generated_code)
```

Using custom sandbox to deeply integrate with your agent system. For instance, you may have your own centralized application states, and your sandbox is just a controller (in the sense of MVC) to access them.

## Finite Python (FPy)

FPy is the restricted Python subset interpreted by Sandboxed Python. It supports:
- Expressions: Constants, variables, binary/unary/comparison ops, calls, indexing, lists/dicts/sets/tuples with starred unpacking, slices.
- Statements: Assignments (`lhs1 = lhs2 = ...`, `e1[e2] = e3`), if-else, try-except-finally (single handler), delete, pass.
- No loops, functions, classes, imports, or comprehensions for security and finiteness.

For the full abstract syntax tree (AST) structure, see [fpy.asdl](https://github.com/fixed-ai/sandboxed-python/blob/main/fpy.asdl).

## Alternatives

- **WASM-based Python Environments**: Projects like Pyodide or CPython-on-Wasm provide browser-based execution but require WebAssembly runtimes and have higher overhead. Sandboxed Python is lighter, pure-Python, and focused on server-side/embedded use cases.
- **RestrictedPython**: A policy-based restrictor for CPython, but... you know...
- Container-based solutions: Not strict enough, and not interop-friendly. Containers are not for sandboxing.

## Deviations from Strict CPython Subset

FPy is mostly a subset of CPython, but with intentional deviations for security:

- **No Direct Access to Exceptions**: In CPython, `except Exception as e:` gives direct access to the exception object, which could expose sensitive internals (e.g., stack traces with paths). In FPy, we prevent this to avoid information leaks in untrusted environments.

  **Escape Hatch**: If you need exception details, the bound variable (e.g., `e`) is a dictionary with safe, string-only keys:
  - `'type'`: A string representing the exception type (e.g., `'ZeroDivisionError'`).
  - `'message'`: Human-readable message.
  - `'py_stacktrace'`: A string representing the original Python backtrace (can be configured to be empty for security).

  Example:
  ```python
  try:
      1 / 0
  except Exception as e:
      print(e['type'])  # Outputs: 'ZeroDivisionError'
  ```

This balances usability with security, ensuring agents/LLMs can handle errors without risking exploits.

## Misc

- **License**: BSD-3-Clause
- **Contributing**:
  - Improve [tests](src/sandboxed_python/tests/) by adding more edge cases or coverage.
  - Improve the documentation/the documentation system, e.g., by expanding examples or generating docs from code.
