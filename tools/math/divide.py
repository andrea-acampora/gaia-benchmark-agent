from langchain_core.tools import tool


@tool
def divide(a: int, b: int) -> int:
    """Divide two numbers.

    Args:
        a: first int
        b: second int
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b
