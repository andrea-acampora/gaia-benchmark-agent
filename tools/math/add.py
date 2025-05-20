from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: first int
        b: second int
    """
    return a + b
