from langchain_core.tools import tool
from typing import Optional
import os
from urllib.parse import urlparse
import tempfile


@tool
def save_content_to_file(content: str, filename: Optional[str] = None) -> str:
    """
    Save content to a file and return the path.
    Args:
        content (str): the content to save to the file
        filename (str, optional): the name of the file. If not provided, a random name file will be created.
    """
    temp_dir = tempfile.gettempdir()
    if filename is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        filepath = temp_file.name
    else:
        filepath = os.path.join(temp_dir, filename)

    with open(filepath, "w") as f:
        f.write(content)

    return f"File saved to {filepath}. You can read this file to process its contents."
