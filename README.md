# coolname-eponym

> The creatures themselves that put the 'cool' in coolname.

An ergonomic Python utility that generates evocative, **photorealistic** images
from the adjective-animal slugs produced by the
[`coolname`](https://pypi.org/project/coolname/) package.

Animals are always depicted as *real* animals – never anthropomorphic.  The
adjective is expressed through the animal's physical form, its behaviour, or
the atmosphere of the scene.

> **Example** – `tall-pig` → a pig with distinctly long legs, full body shot,
> National Geographic style.

---

## Installation

```bash
pip install coolname-eponym
```

To use the local **diffusers** backend (GPU recommended):

```bash
pip install "coolname-eponym[diffusers]"
```

---

## Quick start

### CLI

```bash
# Generate an image for a random coolname slug (uses HF Inference API by default)
export HF_TOKEN=hf_...          # free token from huggingface.co/settings/tokens
coolname-eponym generate

# Generate for a specific slug
coolname-eponym generate --slug tall-pig

# Save to a custom path
coolname-eponym generate --slug mystic-fox --output ~/pictures/mystic-fox.png

# Preview the prompt without generating (no API key needed)
coolname-eponym prompt tall-pig

# Dry run – shows slug, prompt and output path
coolname-eponym generate --slug fanatic-hyena --dry-run
```

### Python API

```python
import coolname
from coolname_eponym import build_prompt, generate_image, save_image

# Generate a random slug
parts = coolname.generate(2)          # e.g. ['tall', 'pig']
slug  = "-".join(parts)               # 'tall-pig'

# Build a photorealistic SD prompt
prompt = build_prompt(slug)
# → 'a pig with a distinctly tall physique, full body visible, natural
#    environment, photorealistic wildlife photograph, no anthropomorphism, …'

# Generate the image (requires HF_TOKEN env var for the default hf-api backend)
image = generate_image(prompt, backend="hf-api", seed=42)

# Save it
save_image(image, f"{slug}.png")
```

---

## Backends

| Backend | Requirements | Notes |
|---------|-------------|-------|
| `hf-api` *(default)* | `HF_TOKEN` env var (free) | Cloud inference via Hugging Face |
| `diffusers` | `pip install "coolname-eponym[diffusers]"`, ≥ 6 GB VRAM recommended | Fully local, open-source |

Both backends default to **`stabilityai/stable-diffusion-xl-base-1.0`**.
Override with `--model`:

```bash
# Lighter model for limited hardware
coolname-eponym generate --slug tall-pig \
    --backend diffusers \
    --model runwayml/stable-diffusion-v1-5
```

---

## How prompts are built

Each coolname slug is an `adjective-animal` pair.  The adjective is classified
into one of five semantic categories, each yielding a different prompt strategy:

| Category | Example adjectives | Prompt strategy |
|----------|--------------------|----------------|
| Physical | `tall`, `fat`, `fluffy`, `muscular` | Modifies the animal's body |
| Visual | `radiant`, `dark`, `blazing`, `spotted` | Modifies colouring / light |
| Dynamic | `flying`, `burrowing`, `nocturnal` | Sets the action / motion |
| Atmospheric | `mystic`, `ancient`, `futuristic` | Sets scene and environment |
| Expressive | `brave`, `sly`, `fanatic`, `wise` | Conveys bearing and gaze |

A shared suffix enforces photographic realism and explicitly suppresses
anthropomorphism on every prompt.

---

## CLI reference

```
coolname-eponym generate [OPTIONS]

  --slug SLUG           Adjective-animal slug (random if omitted)
  -o, --output FILE     Output file (default: <slug>.png)
  --backend CHOICE      hf-api | diffusers  [default: hf-api]
  --model HUB_ID        Hugging Face model ID
  --width INT           Image width in pixels  [default: 1024]
  --height INT          Image height in pixels  [default: 1024]
  --steps INT           Diffusion steps  [default: 30]
  --guidance-scale FLT  CFG scale  [default: 7.5]
  --seed INT            Reproducibility seed
  --dry-run             Print prompt/path without generating

coolname-eponym prompt SLUG
  Print the SD prompt for SLUG without generating an image.
```

---

## License

Apache 2.0 – see [LICENSE](LICENSE).
