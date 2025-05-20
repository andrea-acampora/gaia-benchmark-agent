from langchain_core.tools import tool
import cmath


@tool
def square_root(a: float) -> float | complex:
    """
    Get the square root of a number.
    Args:
        a (float): the number to get the square root of
    """
    if a >= 0:
        return a**0.5
    return cmath.sqrt(a)
