"""Command-line entry point for the VLM baseline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .estimator import DEFAULT_MODEL, VLMSizeEstimator

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

DEFAULT_OUTPUT_DIR = Path("results/vlm_baseline")








def resolve_image_path(image_arg: str, data_dir: Path) -> Path:
    """Resolve either an explicit image path or a filename inside data_dir."""

    image_path = Path(image_arg)
    if image_path.exists():
        return image_path

    candidate = data_dir / image_arg
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Could not find image '{image_arg}' or '{candidate}'")


def list_images(data_dir: Path) -> list[Path]:
    """List supported image files in the given directory, sorted alphabetically."""
    
    return sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Estimate the target object's top-down height and width from an image."
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Image path, filename under --data-dir, or omitted to use the first image in --data-dir.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Directory used when resolving image filenames. Default: data",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("VLM_BASELINE_MODEL", DEFAULT_MODEL),
        help=f"OpenAI vision model. Default: env VLM_BASELINE_MODEL or {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. Default: results/vlm_baseline/<image_stem>_vlm_baseline.json",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Default output directory when --output is omitted. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser






def main() -> None:
    args = build_parser().parse_args()

    if args.image:
        image_path = resolve_image_path(args.image, args.data_dir)
    else:
        images = list_images(args.data_dir)
        if not images:
            raise FileNotFoundError(
                f"No supported images found in {args.data_dir}. "
                f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
            )
        image_path = images[0]

    estimate = VLMSizeEstimator(model=args.model).estimate(image_path)
    payload = estimate.to_dict()
    rendered = json.dumps(payload, indent=2, sort_keys=True)

    output_path = args.output or args.output_dir / f"{image_path.stem}_vlm_baseline.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered + "\n", encoding="utf-8")

    print(rendered)


if __name__ == "__main__":
    main()
