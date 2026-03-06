"""
Command-line interface for coolname-eponym.

Examples
--------
Generate an image for a random coolname slug::

    coolname-eponym generate

Generate an image for a specific slug::

    coolname-eponym generate --slug tall-pig

Just preview the prompt without generating an image::

    coolname-eponym prompt tall-pig

Use the local diffusers backend with a smaller model::

    coolname-eponym generate \\
        --backend diffusers \\
        --model runwayml/stable-diffusion-v1-5 \\
        --steps 20
"""

from __future__ import annotations

import pathlib
import sys

import click
import coolname

from .generator import generate_image, save_image
from .prompt import build_prompt, build_prompt_from_parts


@click.group()
def main() -> None:  # pragma: no cover
    """coolname-eponym – evocative animal images from coolname slugs."""


# ---------------------------------------------------------------------------
# `generate` subcommand
# ---------------------------------------------------------------------------

@main.command()
@click.option(
    "--slug",
    default=None,
    metavar="SLUG",
    help=(
        "Adjective-animal slug to visualise (e.g. 'tall-pig'). "
        "If omitted, a random slug is generated with coolname."
    ),
)
@click.option(
    "--output",
    "-o",
    default=None,
    metavar="FILE",
    help=(
        "Output file path. Defaults to '<slug>.png' in the current directory."
    ),
)
@click.option(
    "--backend",
    default="hf-api",
    type=click.Choice(["hf-api", "diffusers"], case_sensitive=False),
    show_default=True,
    help="Image-generation backend to use.",
)
@click.option(
    "--model",
    default=None,
    metavar="HUB_ID",
    help=(
        "Hugging Face model repository ID. "
        "Defaults to stabilityai/stable-diffusion-xl-base-1.0."
    ),
)
@click.option(
    "--width",
    default=1024,
    type=int,
    show_default=True,
    help="Image width in pixels.",
)
@click.option(
    "--height",
    default=1024,
    type=int,
    show_default=True,
    help="Image height in pixels.",
)
@click.option(
    "--steps",
    default=30,
    type=int,
    show_default=True,
    help="Number of diffusion steps (higher = better quality, slower).",
)
@click.option(
    "--guidance-scale",
    default=7.5,
    type=float,
    show_default=True,
    help="CFG guidance scale.",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    metavar="INT",
    help="Random seed for reproducible generation.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the prompt and output path without generating an image.",
)
def generate(
    slug: str | None,
    output: str | None,
    backend: str,
    model: str | None,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int | None,
    dry_run: bool,
) -> None:
    """Generate an evocative, photorealistic image from a coolname slug."""
    # Resolve slug
    if slug is None:
        parts = coolname.generate(2)
        slug = "-".join(parts)
        click.echo(f"Generated slug: {slug}")

    # Build prompt
    prompt = build_prompt(slug)
    click.echo(f"Prompt: {prompt}")

    # Resolve output path
    if output is None:
        output = f"{slug}.png"
    output_path = pathlib.Path(output)

    click.echo(f"Output: {output_path}")

    if dry_run:
        click.echo("Dry run – no image generated.")
        return

    click.echo(f"Generating with backend={backend!r}…")
    try:
        image = generate_image(
            prompt,
            backend=backend,
            model=model,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
    except RuntimeError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    saved = save_image(image, output_path)
    click.echo(f"Saved to {saved}")


# ---------------------------------------------------------------------------
# `prompt` subcommand – preview a prompt without generating an image
# ---------------------------------------------------------------------------

@main.command("prompt")
@click.argument("slug")
def show_prompt(slug: str) -> None:
    """Print the image-generation prompt for SLUG without generating an image.

    SLUG should be an adjective-animal pair separated by a hyphen,
    e.g. 'tall-pig' or 'mystic-fox'.
    """
    prompt = build_prompt(slug)
    click.echo(prompt)
