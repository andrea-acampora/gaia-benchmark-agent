from langchain_core.tools import tool
from typing import Dict, Any
import base64
from PIL import Image
import io
import os
import uuid
import numpy as np

# Helper functions for image processing


def decode_image(base64_string: str) -> Image.Image:
    """Convert a base64 string to a PIL Image."""
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))


def save_image(image: Image.Image, directory: str = "image_outputs") -> str:
    """Save a PIL Image to disk and return the path."""
    os.makedirs(directory, exist_ok=True)
    image_id = str(uuid.uuid4())
    image_path = os.path.join(directory, f"{image_id}.png")
    image.save(image_path)
    return image_path


def encode_image(image_path: str) -> str:
    """Convert an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@tool
def analyze_image(image_base64: str) -> Dict[str, Any]:
    """
    Analyze basic properties of an image (size, mode, color analysis, thumbnail preview).
    Args:
        image_base64 (str): Base64 encoded image string
    Returns:
        Dictionary with analysis result
    """
    try:
        img = decode_image(image_base64)
        width, height = img.size
        mode = img.mode

        if mode in ("RGB", "RGBA"):
            arr = np.array(img)
            avg_colors = arr.mean(axis=(0, 1))
            dominant = ["Red", "Green", "Blue"][np.argmax(avg_colors[:3])]
            brightness = avg_colors.mean()
            color_analysis = {
                "average_rgb": avg_colors.tolist(),
                "brightness": brightness,
                "dominant_color": dominant,
            }
        else:
            color_analysis = {"note": f"No color analysis for mode {mode}"}

        thumbnail = img.copy()
        thumbnail.thumbnail((100, 100))
        thumb_path = save_image(thumbnail, "thumbnails")
        thumbnail_base64 = encode_image(thumb_path)

        return {
            "dimensions": (width, height),
            "mode": mode,
            "color_analysis": color_analysis,
            "thumbnail": thumbnail_base64,
        }
    except Exception as e:
        return {"error": str(e)}
