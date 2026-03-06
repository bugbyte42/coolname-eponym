"""Tests for coolname_eponym.generator."""

import io
import pathlib

import pytest
from PIL import Image
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_image() -> Image.Image:
    """Return a tiny RGB image for use in mocks."""
    return Image.new("RGB", (64, 64), color=(42, 100, 200))


# ---------------------------------------------------------------------------
# generate_image – hf-api backend
# ---------------------------------------------------------------------------


class TestGenerateImageHfApi:
    def test_calls_requests_post(self, monkeypatch, tmp_path):
        """generate_image with hf-api calls requests.post and returns an Image."""
        fake_img = _make_fake_image()
        buf = io.BytesIO()
        fake_img.save(buf, format="PNG")
        buf.seek(0)
        raw_bytes = buf.read()

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.content = raw_bytes

        monkeypatch.setenv("HF_TOKEN", "hf_test_token")

        with patch("coolname_eponym.generator.requests.post", return_value=mock_response) as mock_post:
            from coolname_eponym.generator import generate_image

            result = generate_image(
                "a tall pig",
                backend="hf-api",
                model="test/model",
            )

        assert isinstance(result, Image.Image)
        mock_post.assert_called_once()

    def test_raises_without_hf_token(self, monkeypatch):
        """RuntimeError raised when HF_TOKEN is missing."""
        monkeypatch.delenv("HF_TOKEN", raising=False)

        from coolname_eponym.generator import generate_image

        with pytest.raises(RuntimeError, match="HF_TOKEN"):
            generate_image("a test prompt", backend="hf-api")

    def test_raises_on_503(self, monkeypatch):
        """RuntimeError with friendly message on model-loading 503."""
        monkeypatch.setenv("HF_TOKEN", "hf_test_token")

        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 503

        with patch("coolname_eponym.generator.requests.post", return_value=mock_response):
            from coolname_eponym.generator import generate_image

            with pytest.raises(RuntimeError, match="loading"):
                generate_image("a test prompt", backend="hf-api")

    def test_raises_on_non_ok_response(self, monkeypatch):
        """RuntimeError raised for non-200 responses."""
        monkeypatch.setenv("HF_TOKEN", "hf_test_token")

        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch("coolname_eponym.generator.requests.post", return_value=mock_response):
            from coolname_eponym.generator import generate_image

            with pytest.raises(RuntimeError, match="401"):
                generate_image("a test prompt", backend="hf-api")


# ---------------------------------------------------------------------------
# generate_image – diffusers backend
# ---------------------------------------------------------------------------


class TestGenerateImageDiffusers:
    def test_raises_importerror_if_not_installed(self, monkeypatch):
        """ImportError with install hint when diffusers is absent."""
        # Simulate diffusers not being installed
        import sys
        original_torch = sys.modules.get("torch")
        original_diffusers = sys.modules.get("diffusers")

        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(sys.modules, "diffusers", None)

        # We need to re-import the function to pick up the mocked modules
        import importlib
        import coolname_eponym.generator as gen_mod
        importlib.reload(gen_mod)

        with pytest.raises(ImportError, match="diffusers"):
            gen_mod._generate_diffusers(
                "a prompt",
                model="test/model",
                negative_prompt="",
                width=64,
                height=64,
                steps=1,
                guidance_scale=7.5,
                seed=None,
            )

        # Restore
        if original_torch is None:
            monkeypatch.delitem(sys.modules, "torch", raising=False)
        else:
            monkeypatch.setitem(sys.modules, "torch", original_torch)
        if original_diffusers is None:
            monkeypatch.delitem(sys.modules, "diffusers", raising=False)
        else:
            monkeypatch.setitem(sys.modules, "diffusers", original_diffusers)
        importlib.reload(gen_mod)


# ---------------------------------------------------------------------------
# generate_image – unknown backend
# ---------------------------------------------------------------------------


class TestGenerateImageUnknownBackend:
    def test_raises_valueerror(self):
        from coolname_eponym.generator import generate_image

        with pytest.raises(ValueError, match="Unknown backend"):
            generate_image("a prompt", backend="telepathy")


# ---------------------------------------------------------------------------
# save_image
# ---------------------------------------------------------------------------


class TestSaveImage:
    def test_saves_png(self, tmp_path):
        from coolname_eponym.generator import save_image

        img = _make_fake_image()
        dest = tmp_path / "output.png"
        result = save_image(img, dest)

        assert result == dest.resolve()
        assert dest.exists()
        saved = Image.open(dest)
        assert saved.size == (64, 64)

    def test_creates_parent_directories(self, tmp_path):
        from coolname_eponym.generator import save_image

        img = _make_fake_image()
        dest = tmp_path / "a" / "b" / "c" / "output.png"
        save_image(img, dest)
        assert dest.exists()

    def test_returns_resolved_path(self, tmp_path):
        from coolname_eponym.generator import save_image

        img = _make_fake_image()
        dest = tmp_path / "out.png"
        result = save_image(img, dest)
        assert isinstance(result, pathlib.Path)
        assert result.is_absolute()
