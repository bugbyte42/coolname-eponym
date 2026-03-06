"""Tests for coolname_eponym.cli."""

import io
import pathlib

import pytest
from click.testing import CliRunner
from PIL import Image
from unittest.mock import MagicMock, patch

from coolname_eponym.cli import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_image() -> Image.Image:
    return Image.new("RGB", (64, 64), color=(10, 20, 30))


# ---------------------------------------------------------------------------
# `generate` subcommand – dry run
# ---------------------------------------------------------------------------


class TestGenerateCommand:
    def test_dry_run_with_slug(self):
        runner = CliRunner()
        result = runner.invoke(main, ["generate", "--slug", "tall-pig", "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "tall-pig" in result.output
        assert "pig" in result.output.lower()
        assert "Dry run" in result.output

    def test_dry_run_random_slug(self):
        runner = CliRunner()
        result = runner.invoke(main, ["generate", "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "Generated slug:" in result.output
        assert "Prompt:" in result.output

    def test_dry_run_shows_output_path(self, tmp_path):
        runner = CliRunner()
        out = str(tmp_path / "my-pig.png")
        result = runner.invoke(
            main, ["generate", "--slug", "tall-pig", "--dry-run", "--output", out]
        )
        assert result.exit_code == 0, result.output
        assert "my-pig.png" in result.output

    def test_dry_run_default_output_uses_slug(self):
        runner = CliRunner()
        result = runner.invoke(main, ["generate", "--slug", "tall-pig", "--dry-run"])
        assert "tall-pig.png" in result.output

    def test_generate_calls_backend(self, monkeypatch, tmp_path):
        """Full generate path calls generate_image and save_image."""
        fake_img = _make_fake_image()
        monkeypatch.setenv("HF_TOKEN", "hf_test")

        out = str(tmp_path / "out.png")

        with patch("coolname_eponym.cli.generate_image", return_value=fake_img) as mock_gen, \
             patch("coolname_eponym.cli.save_image", return_value=pathlib.Path(out)) as mock_save:
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["generate", "--slug", "tall-pig", "--output", out],
            )

        assert result.exit_code == 0, result.output
        mock_gen.assert_called_once()
        mock_save.assert_called_once()

    def test_generate_error_exits_nonzero(self, monkeypatch, tmp_path):
        """A RuntimeError from the backend exits with code 1."""
        monkeypatch.setenv("HF_TOKEN", "hf_test")

        with patch(
            "coolname_eponym.cli.generate_image",
            side_effect=RuntimeError("model error"),
        ):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["generate", "--slug", "tall-pig", "--output", str(tmp_path / "out.png")],
            )

        assert result.exit_code == 1

    def test_generate_accepts_backend_choice(self):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["generate", "--slug", "tall-pig", "--backend", "diffusers", "--dry-run"],
        )
        assert result.exit_code == 0, result.output

    def test_generate_rejects_invalid_backend(self):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["generate", "--slug", "tall-pig", "--backend", "telepathy", "--dry-run"],
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# `prompt` subcommand
# ---------------------------------------------------------------------------


class TestPromptCommand:
    def test_prints_prompt_for_slug(self):
        runner = CliRunner()
        result = runner.invoke(main, ["prompt", "tall-pig"])
        assert result.exit_code == 0, result.output
        assert "pig" in result.output.lower()
        assert "tall" in result.output.lower()

    def test_prompt_is_photorealistic(self):
        runner = CliRunner()
        result = runner.invoke(main, ["prompt", "mystic-fox"])
        assert "photorealistic" in result.output.lower()

    def test_prompt_for_random_slugs(self):
        """Spot-check several slugs via the CLI."""
        runner = CliRunner()
        for slug in ["fierce-lion", "tiny-sparrow", "ancient-tortoise"]:
            result = runner.invoke(main, ["prompt", slug])
            assert result.exit_code == 0, f"Failed for slug {slug!r}: {result.output}"
            animal = slug.split("-", 1)[1]
            assert animal in result.output.lower()
