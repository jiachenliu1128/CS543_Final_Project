# CS543_Final_Project

## VLM baseline

The `src/vlm_baseline` module estimates the top-down physical size of the object centered in a single image, without using a reference object. It uses an API vision-language model with structured JSON output.

Setup:

```bash
pip install -r requirements.txt
```

Create `.env` in the repo root:

```bash
OPENAI_API_KEY=...
```

Run on an image in `data/`:

```bash
python scripts/run_vlm_baseline.py example.jpg
```

Run on an explicit path:

```bash
python scripts/run_vlm_baseline.py data/example.jpg
```

By default, results are saved to `results/vlm_baseline/<image_stem>_vlm_baseline.json`. Use `--output` to choose an exact output path.

Output format:

```json
{
  "height_cm": 14.0,
  "width_cm": 7.2
}
```

The default model is `gpt-5.4-mini`, a cheaper modern OpenAI vision model. Override it with `--model` or `VLM_BASELINE_MODEL`.
