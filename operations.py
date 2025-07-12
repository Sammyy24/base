from langchain_core.tools import tool

@tool
def add(a: float, b: float) -> str:
    """Add two numbers."""
    return f"The result of {a} + {b} is {a + b}"

@tool
def subtract(a: float, b: float) -> str:
    """Subtract b from a."""
    return f"The result of {a} - {b} is {a - b}"

@tool
def multiply(a: float, b: float) -> str:
    """Multiply two numbers."""
    return f"The result of {a} * {b} is {a * b}"
