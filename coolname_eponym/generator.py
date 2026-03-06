"""
Image generation backends for coolname-eponym.

Two backends are supported:

hf-api (default)
    Uses the Hugging Face Inference API – no local GPU required.  Requires
    the environment variable ``HF_TOKEN`` (a free HuggingFace account token).
    Model defaults to ``stabilityai/stable-diffusion-xl-base-1.0``.

diffusers
    Runs a Stable Diffusion model locally via the ``diffusers`` library.
    Requires ``pip install "coolname-eponym[diffusers]"`` and sufficient
    VRAM (≥ 6 GB for SDXL, or use ``--model runwayml/stable-diffusion-v1-5``
    for lighter hardware).  Falls back to CPU automatically.
"""

from __future__ import annotations

import io
import os
import pathlib
from typing import Optional

import requests
from PIL import Image

from .prompt import NEGATIVE_PROMPT

# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------
DEFAULT_HF_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_DIFFUSERS_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE = 7.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hf_api_token() -> str:
    """Return the HF token from the environment or raise a clear error."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "The HF_TOKEN environment variable is not set.\n"
            "Create a free token at https://huggingface.co/settings/tokens "
            "and export it:\n\n"
            "    export HF_TOKEN=hf_...\n"
        )
    return token


# ---------------------------------------------------------------------------
# Public generate function
# ---------------------------------------------------------------------------

def generate_image(
    prompt: str,
    *,
    backend: str = "hf-api",
    model: Optional[str] = None,
    negative_prompt: str = NEGATIVE_PROMPT,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    steps: int = DEFAULT_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE,
    seed: Optional[int] = None,
) -> Image.Image:
    """Generate an image from *prompt* and return a :class:`PIL.Image.Image`.

    Parameters
    ----------
    prompt:
        The full Stable Diffusion prompt (typically from
        :func:`coolname_eponym.prompt.build_prompt`).
    backend:
        ``"hf-api"`` (default) or ``"diffusers"``.
    model:
        Hub model ID.  Defaults to ``stabilityai/stable-diffusion-xl-base-1.0``
        for both backends.
    negative_prompt:
        What the model should *not* produce.
    width / height:
        Image dimensions in pixels.  SDXL works best at 1024×1024.
    steps:
        Number of diffusion steps (higher → better quality, slower).
    guidance_scale:
        CFG scale – how strongly the prompt guides generation.
    seed:
        Optional RNG seed for reproducibility.
    """
    if backend == "hf-api":
        return _generate_hf_api(
            prompt,
            model=model or DEFAULT_HF_MODEL,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
    if backend == "diffusers":
        return _generate_diffusers(
            prompt,
            model=model or DEFAULT_DIFFUSERS_MODEL,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
    raise ValueError(
        f"Unknown backend {backend!r}. Choose 'hf-api' or 'diffusers'."
    )


# ---------------------------------------------------------------------------
# HF Inference API backend
# ---------------------------------------------------------------------------

def _generate_hf_api(
    prompt: str,
    *,
    model: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: Optional[int],
) -> Image.Image:
    """Call the Hugging Face Inference API and return the generated image."""
    token = _hf_api_token()
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    payload: dict = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
        },
    }
    if seed is not None:
        payload["parameters"]["seed"] = seed

    response = requests.post(url, headers=headers, json=payload, timeout=120)

    if response.status_code == 503:
        raise RuntimeError(
            f"Model {model!r} is still loading on the HF servers. "
            "Please wait a moment and try again."
        )
    if not response.ok:
        raise RuntimeError(
            f"HF Inference API returned {response.status_code}: "
            f"{response.text[:200]}"
        )

    return Image.open(io.BytesIO(response.content)).convert("RGB")


# ---------------------------------------------------------------------------
# Local diffusers backend
# ---------------------------------------------------------------------------

def _generate_diffusers(
    prompt: str,
    *,
    model: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: Optional[int],
) -> Image.Image:
    """Run a diffusion pipeline locally via the *diffusers* library."""
    try:
        import torch
        from diffusers import DiffusionPipeline
    except ImportError as exc:
        raise ImportError(
            "The 'diffusers' backend requires additional packages.\n"
            "Install them with:\n\n"
            "    pip install \"coolname-eponym[diffusers]\"\n"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = DiffusionPipeline.from_pretrained(
        model,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe = pipe.to(device)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    return result.images[0]


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_image(image: Image.Image, path: str | pathlib.Path) -> pathlib.Path:
    """Save *image* to *path* (creating parent directories as needed).

    Parameters
    ----------
    image:
        PIL image returned by :func:`generate_image`.
    path:
        Destination file path.  The file format is inferred from the
        extension; ``.png`` and ``.jpg`` / ``.jpeg`` are most common.

    Returns
    -------
    pathlib.Path
        The resolved output path.
    """
    out = pathlib.Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    return out
