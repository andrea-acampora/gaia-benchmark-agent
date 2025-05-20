from langchain_core.tools import tool


@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers.

    Args:
        a: first int
        b: second int
    """
    return a - b
