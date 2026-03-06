"""
Prompt engineering: convert a coolname adjective-animal slug into a
Stable Diffusion image prompt that is photorealistic and non-anthropomorphic.

The prompt always depicts the *real* animal.  The adjective is rendered as a
physical trait, a behavioral pose, or a scene/atmosphere – never as clothing,
a human expression, or any humanising device.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Adjective classification
# ---------------------------------------------------------------------------
# Each set contains adjectives that share the same mapping strategy.

# PHYSICAL – adjective maps to a bodily attribute of the animal.
_PHYSICAL: frozenset[str] = frozenset(
    [
        "amorphous", "armored", "bald", "bipedal", "boisterous",
        "bouncy", "brawny", "broad", "bulky", "caped", "chubby",
        "curly", "curvy", "dainty", "elastic", "fat", "feathered",
        "flat", "fluffy", "foamy", "fragrant", "furry", "fuzzy",
        "glaring", "glossy", "hairy", "heavy", "horned", "juicy",
        "lean", "light", "loose", "lumpy", "masked", "meaty",
        "mottled", "muscular", "opalescent", "rough", "ruddy",
        "shaggy", "shapeless", "shiny", "silky", "skinny", "slick",
        "slim", "smooth", "soft", "spiked", "spotted", "sticky",
        "striped", "tall", "tentacled", "thick", "thin", "translucent",
        "transparent", "wooden",
    ]
)

# VISUAL / LIGHT – adjective describes how the animal looks or how light plays on it.
_VISUAL: frozenset[str] = frozenset(
    [
        "blazing", "bright", "brilliant", "camouflaged", "colorful",
        "crystal", "dark", "dazzling", "electric", "ethereal",
        "fiery", "flashy", "fluorescent", "gleaming", "gleeful",
        "glistening", "glittering", "glorious", "golden", "hidden",
        "hypnotic", "icy", "luminous", "misty", "neon", "pastel",
        "psychedelic", "radiant", "shiny", "silky", "smoky",
        "sparkling", "spectral", "vivid",
    ]
)

# DYNAMIC / ACTION – adjective maps to a physical motion or behaviour.
_DYNAMIC: frozenset[str] = frozenset(
    [
        "active", "agile", "airborne", "amphibian", "aquatic",
        "arboreal", "athletic", "burrowing", "crouching", "dancing",
        "flying", "hopping", "hissing", "jumping", "lurking",
        "nocturnal", "purring", "roaring", "running", "screeching",
        "singing", "stalking", "swinging", "terrestrial",
        "thundering", "tunneling", "wandering", "weightless",
        "winged",
    ]
)

# ATMOSPHERIC / SCENE – adjective sets the mood, setting, or environment.
_ATMOSPHERIC: frozenset[str] = frozenset(
    [
        "aboriginal", "ancient", "angelic", "antique", "arcane",
        "archetypal", "artificial", "auspicious", "augmented",
        "celestial", "classic", "cryptic", "demonic", "dramatic",
        "enchanted", "esoteric", "fictional", "futuristic",
        "godlike", "hallowed", "heavenly", "heretic", "holistic",
        "imaginary", "immortal", "imperial", "legendary", "lush",
        "military", "modern", "monumental", "mystical", "mystic",
        "natural", "nebulous", "organic", "poetic", "polar",
        "prehistoric", "primitive", "prophetic", "quantum",
        "romantic", "rustic", "spectral", "spiritual", "urban",
        "utopian", "venerable",
    ]
)

# EXPRESSIVE – personality/character rendered via pose, expression, or gaze.
_EXPRESSIVE: frozenset[str] = frozenset(
    [
        "abiding", "able", "abstract", "academic", "accomplished",
        "accurate", "adamant", "adaptable", "adept", "admirable",
        "adorable", "adventurous", "affable", "aggressive", "aloof",
        "amazing", "ambitious", "amiable", "amusing", "analytic",
        "arrogant", "aspiring", "astonishing", "astute", "attentive",
        "attractive", "audacious", "authentic", "axiomatic",
        "beautiful", "belligerent", "benevolent", "benign", "berserk",
        "bizarre", "bold", "brainy", "brave", "busy", "calculating",
        "calm", "capable", "careful", "casual", "cautious", "certain",
        "charming", "cheerful", "cherubic", "chirpy", "chivalrous",
        "clever", "cocky", "comical", "competent", "congenial",
        "controversial", "convivial", "cooperative", "cordial",
        "courageous", "crafty", "crazy", "cunning", "curious",
        "cute", "daffy", "daft", "dangerous", "daring", "dashing",
        "debonair", "defiant", "deft", "delectable", "delicate",
        "delightful", "determined", "devious", "devout", "dexterous",
        "diligent", "discerning", "discreet", "divergent", "dynamic",
        "eager", "eccentric", "easygoing", "ecstatic", "effective",
        "efficient", "elated", "elegant", "elusive", "enigmatic",
        "enthusiastic", "exceptional", "exotic", "exuberant",
        "fabulous", "fair", "faithful", "famous", "fanatic",
        "fantastic", "fascinating", "fearless", "fervent", "festive",
        "fierce", "fine", "finicky", "flawless", "flexible",
        "formidable", "fortunate", "friendly", "frisky", "funny",
        "furious", "gainful", "garrulous", "generous", "gentle",
        "gifted", "girlish", "graceful", "gracious", "grateful",
        "greedy", "gregarious", "grinning", "groovy", "grumpy",
        "handsome", "happy", "hasty", "healthy", "helpful",
        "hilarious", "honest", "honorable", "honored", "hopeful",
        "hospitable", "hot", "humble", "humorous", "hungry",
        "hysterical", "idealistic", "illustrious", "impartial",
        "impetuous", "independent", "industrious", "inescapable",
        "infallible", "influential", "ingenious", "innocent",
        "inquisitive", "inscrutable", "intelligent", "interesting",
        "intrepid", "invaluable", "inventive", "invincible",
        "invisible", "jolly", "jovial", "judicious", "just", "keen",
        "kind", "knowing", "laughing", "likable", "literate",
        "lively", "logical", "lovely", "loyal", "lucky", "lyrical",
        "macho", "magnificent", "majestic", "manipulative",
        "masterful", "mature", "meek", "mellow", "melodic",
        "memorable", "merciful", "merry", "meticulous", "mighty",
        "mindful", "miraculous", "modest", "musical", "mysterious",
        "naughty", "noble", "nonchalant", "nostalgic", "notorious",
        "obedient", "observant", "offbeat", "omniscient", "optimal",
        "optimistic", "original", "orthodox", "outgoing",
        "outrageous", "outstanding", "overjoyed", "passionate",
        "peculiar", "perfect", "perky", "phenomenal", "placid",
        "poised", "polite", "pompous", "popular", "positive",
        "powerful", "practical", "precious", "proud", "provocative",
        "prudent", "puzzling", "qualified", "quick", "quiet",
        "quirky", "quixotic", "quizzical", "radical", "rare",
        "rational", "real", "realistic", "reasonable", "rebel",
        "refined", "resilient", "resolute", "resourceful",
        "responsible", "rich", "righteous", "rigorous", "robust",
        "rousing", "rugged", "sarcastic", "sassy", "savvy",
        "scrupulous", "secret", "sensible", "serious", "shrewd",
        "simple", "sincere", "skilled", "sloppy", "sly", "smart",
        "smiling", "sociable", "solemn", "solid", "sophisticated",
        "spectacular", "speedy", "spirited", "splendid", "spry",
        "stalwart", "statuesque", "steadfast", "stoic",
        "strange", "strategic", "strict", "strong", "sturdy",
        "stylish", "subtle", "successful", "sweet", "sympathetic",
        "tacky", "tactful", "talented", "talkative", "tangible",
        "tasteful", "terrific", "thankful", "thoughtful", "tireless",
        "tough", "traditional", "tremendous", "tricky", "true",
        "truthful", "unbiased", "unique", "unnatural", "unselfish",
        "unstoppable", "upbeat", "uppish", "uptight", "valiant",
        "vegan", "vehement", "vengeful", "versatile", "victorious",
        "vigilant", "vigorous", "visionary", "vivacious", "vociferous",
        "voracious", "wakeful", "warm", "watchful", "wealthy",
        "whimsical", "wild", "wise", "witty", "wonderful", "wondrous",
        "worthy", "zealous",
    ]
)

# ---------------------------------------------------------------------------
# Per-category prompt fragment templates
# ---------------------------------------------------------------------------
# Each value is a template that can be .format(adjective=adj, animal=animal).

_PHYSICAL_TEMPLATE = (
    "a {animal} with a distinctly {adjective} physique, "
    "full body visible, natural environment"
)

_VISUAL_TEMPLATE = (
    "a {animal} with {adjective} coloration and texture, "
    "natural environment, dramatic natural lighting"
)

_DYNAMIC_TEMPLATE = (
    "a {animal} caught mid-action, {adjective} movement, "
    "wildlife photography, motion blur"
)

_ATMOSPHERIC_TEMPLATE = (
    "a {animal} in a {adjective} landscape, "
    "evocative scene, cinematic composition, golden hour"
)

_EXPRESSIVE_TEMPLATE = (
    "a {animal} with a {adjective} bearing and intense gaze, "
    "striking portrait, natural setting, directional light"
)

# Fallback for adjectives not found in any set.
_FALLBACK_TEMPLATE = (
    "a {adjective} {animal} in its natural habitat, "
    "striking natural pose"
)

# ---------------------------------------------------------------------------
# Shared style suffix added to every prompt.
# ---------------------------------------------------------------------------
_STYLE_SUFFIX = (
    "photorealistic wildlife photograph, "
    "no anthropomorphism, not anthropomorphic, "
    "National Geographic style, "
    "ultra-detailed, 8k resolution, "
    "natural anatomy, real animal"
)

# Negative prompt (returned separately so callers can pass it to the model).
NEGATIVE_PROMPT = (
    "anthropomorphic, cartoon, anime, illustration, painting, "
    "dressed, clothes, human pose, human expression, "
    "digital art, cgi, 3d render, text, watermark, logo"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _classify(adjective: str) -> str:
    """Return the category name for *adjective*."""
    adj = adjective.lower()
    if adj in _PHYSICAL:
        return "physical"
    if adj in _VISUAL:
        return "visual"
    if adj in _DYNAMIC:
        return "dynamic"
    if adj in _ATMOSPHERIC:
        return "atmospheric"
    if adj in _EXPRESSIVE:
        return "expressive"
    return "fallback"


def _adjective_fragment(adjective: str, animal: str) -> str:
    """Build the adjective-specific portion of the prompt."""
    category = _classify(adjective)
    mapping = {
        "physical": _PHYSICAL_TEMPLATE,
        "visual": _VISUAL_TEMPLATE,
        "dynamic": _DYNAMIC_TEMPLATE,
        "atmospheric": _ATMOSPHERIC_TEMPLATE,
        "expressive": _EXPRESSIVE_TEMPLATE,
        "fallback": _FALLBACK_TEMPLATE,
    }
    template = mapping[category]
    return template.format(adjective=adjective, animal=animal)


def build_prompt(slug: str) -> str:
    """Build a Stable Diffusion prompt from an adjective-animal *slug*.

    Parameters
    ----------
    slug:
        A hyphen-joined string produced by :func:`coolname.generate_slug`,
        e.g. ``"tall-pig"`` or ``"mystic-fox"``.  Multi-word animals that
        contain their own hyphens (e.g. ``"ring-tailed-lemur"``) are handled
        correctly: only the first token is treated as the adjective.

    Returns
    -------
    str
        A rich, photorealistic prompt suitable for Stable Diffusion or any
        compatible diffusion model.
    """
    parts = slug.split("-", 1)
    if len(parts) != 2:  # pragma: no cover – defensive
        adjective, animal = slug, "animal"
    else:
        adjective, animal = parts[0], parts[1].replace("-", " ")

    subject = _adjective_fragment(adjective, animal)
    return f"{subject}, {_STYLE_SUFFIX}"


def build_prompt_from_parts(adjective: str, animal: str) -> str:
    """Build a prompt from separate *adjective* and *animal* strings.

    Convenience wrapper around :func:`build_prompt` for callers that already
    have the slug split into parts (e.g. from ``coolname.generate(2)``).
    """
    # Normalise multi-word animals (e.g. ["ring", "tailed", "lemur"] joined)
    animal_str = animal.replace("-", " ")
    return build_prompt(f"{adjective}-{animal_str}")
