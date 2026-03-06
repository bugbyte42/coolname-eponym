"""
coolname-eponym – evocative animal images from coolname adjective-animal slugs.

Public API
----------
.. code-block:: python

    from coolname_eponym import build_prompt, generate_image, save_image

    slug = "tall-pig"
    prompt = build_prompt(slug)
    image = generate_image(prompt, backend="hf-api")
    save_image(image, "tall-pig.png")
"""

from .generator import generate_image, save_image
from .prompt import NEGATIVE_PROMPT, build_prompt, build_prompt_from_parts

__all__ = [
    "build_prompt",
    "build_prompt_from_parts",
    "generate_image",
    "save_image",
    "NEGATIVE_PROMPT",
]
