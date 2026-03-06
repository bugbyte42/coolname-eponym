"""Tests for coolname_eponym.prompt."""

import pytest

from coolname_eponym.prompt import (
    NEGATIVE_PROMPT,
    _classify,
    build_prompt,
    build_prompt_from_parts,
)

# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_returns_string(self):
        result = build_prompt("tall-pig")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_animal(self):
        result = build_prompt("tall-pig")
        assert "pig" in result.lower()

    def test_contains_adjective(self):
        result = build_prompt("tall-pig")
        assert "tall" in result.lower()

    def test_photorealism_suffix_present(self):
        result = build_prompt("tall-pig")
        assert "photorealistic" in result.lower()

    def test_no_anthropomorphism_in_prompt(self):
        result = build_prompt("tall-pig")
        assert "no anthropomorphism" in result.lower()

    def test_multi_word_animal(self):
        """Hyphens inside the animal portion are converted to spaces."""
        result = build_prompt("ancient-ring-tailed-lemur")
        assert "ring tailed lemur" in result.lower()

    def test_physical_adjective_physical_category(self):
        """'tall' is a physical adjective – body description expected."""
        result = build_prompt("tall-pig")
        # Physical template uses 'physique'
        assert "physique" in result.lower()

    def test_atmospheric_adjective(self):
        result = build_prompt("mystic-fox")
        assert "landscape" in result.lower() or "mystic" in result.lower()

    def test_dynamic_adjective(self):
        result = build_prompt("flying-eagle")
        assert "action" in result.lower() or "motion" in result.lower()

    def test_expressive_adjective(self):
        result = build_prompt("fierce-lion")
        assert "bearing" in result.lower() or "gaze" in result.lower()

    def test_visual_adjective(self):
        result = build_prompt("radiant-peacock")
        assert "coloration" in result.lower() or "texture" in result.lower()

    def test_fallback_for_unknown_adjective(self):
        result = build_prompt("xyzzy-elephant")
        assert "elephant" in result.lower()
        assert "xyzzy" in result.lower()

    def test_slug_with_only_one_part_handled(self):
        """If the slug has no hyphen the whole string is the animal."""
        result = build_prompt("elephant")
        assert "animal" in result.lower() or "elephant" in result.lower()

    def test_many_real_slugs_include_animal(self):
        """Spot-check a variety of real coolname combinations."""
        cases = [
            ("fat-bear", "bear"),
            ("fluffy-rabbit", "rabbit"),
            ("dark-wolf", "wolf"),
            ("brave-eagle", "eagle"),
            ("ancient-tortoise", "tortoise"),
            ("electric-eel", "eel"),
            ("burrowing-owl", "owl"),
        ]
        for slug, animal in cases:
            result = build_prompt(slug)
            assert animal in result.lower(), f"{animal} missing in prompt for {slug!r}"


# ---------------------------------------------------------------------------
# build_prompt_from_parts
# ---------------------------------------------------------------------------


class TestBuildPromptFromParts:
    def test_equivalent_to_build_prompt(self):
        assert build_prompt_from_parts("tall", "pig") == build_prompt("tall-pig")

    def test_multi_word_animal(self):
        result = build_prompt_from_parts("ancient", "ring-tailed-lemur")
        assert "ring tailed lemur" in result.lower()


# ---------------------------------------------------------------------------
# _classify
# ---------------------------------------------------------------------------


class TestClassify:
    def test_physical(self):
        assert _classify("tall") == "physical"
        assert _classify("fat") == "physical"
        assert _classify("fluffy") == "physical"

    def test_visual(self):
        assert _classify("radiant") == "visual"
        assert _classify("dark") == "visual"

    def test_dynamic(self):
        assert _classify("flying") == "dynamic"
        assert _classify("burrowing") == "dynamic"
        assert _classify("nocturnal") == "dynamic"

    def test_atmospheric(self):
        assert _classify("mystic") == "atmospheric"
        assert _classify("ancient") == "atmospheric"
        assert _classify("futuristic") == "atmospheric"

    def test_expressive(self):
        assert _classify("brave") == "expressive"
        assert _classify("fierce") == "expressive"
        assert _classify("wise") == "expressive"

    def test_fallback(self):
        assert _classify("xyzzy") == "fallback"
        assert _classify("nonexistentword") == "fallback"

    def test_case_insensitive(self):
        assert _classify("TALL") == "physical"
        assert _classify("Mystic") == "atmospheric"


# ---------------------------------------------------------------------------
# NEGATIVE_PROMPT sanity check
# ---------------------------------------------------------------------------


class TestNegativePrompt:
    def test_contains_key_exclusions(self):
        neg = NEGATIVE_PROMPT.lower()
        assert "anthropomorphic" in neg
        assert "cartoon" in neg
        assert "clothes" in neg
