"""Utility functions for image processing and data handling (app copy).

This is the same code as the top-level `utils.py` but available under
`app.utils` for package-style imports.
"""
import base64
import io
from PIL import Image
import requests
from app.logger import get_logger

log = get_logger("utils")


def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    img = Image.open(io.BytesIO(response.content))
    log.debug("loaded_image_from_url", extra={"url": url, "size": img.size})
    return img


def load_image_from_base64(base64_string: str) -> Image.Image:
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    image_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(image_data))
    log.debug("loaded_image_from_base64", extra={"size": img.size})
    return img


def load_image(image_input: str) -> Image.Image:
    if image_input.startswith(('http://', 'https://')):
        return load_image_from_url(image_input)
    else:
        return load_image_from_base64(image_input)

