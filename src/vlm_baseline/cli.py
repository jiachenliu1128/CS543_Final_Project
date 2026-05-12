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








def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Estimate the target object's top-down height and width from an image."
    )
    parser.add_argument(
        "image",
        nargs="?",
        help=(
            "Image path, filename under --data_dir, 'all' to process every image in "
            "--data_dir, or omitted to use the first image in --data_dir."
        ),
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/543 images"),
        help="Directory used when resolving image filenames. Default: data/543 images",
    )
    parser.add_argument(
        "--csv_dir",
        type=Path,
        default=Path("data/cs543_data_collection_plan_updated.csv"),
        help="Path to the CSV metadata file. Default: data/cs543_data_collection_plan_updated.csv",
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




## Helper functions for resolving images, listing directories, estimating and writing results, and building output paths.

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


def estimate_and_write(
    estimator: VLMSizeEstimator,
    image_path: Path,
    output_path: Path,
) -> dict[str, float]:
    """Estimate one image and write the JSON payload."""

    estimate = estimator.estimate(image_path)
    payload = estimate.to_dict()
    rendered = json.dumps(payload, indent=2, sort_keys=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered + "\n", encoding="utf-8")

    return payload


def default_output_path(image_path: Path, output_dir: Path) -> Path:
    """Build the default output path for one image."""

    return output_dir / f"{image_path.stem}_vlm_baseline.json"


def get_images_or_raise(data_dir: Path) -> list[Path]:
    """List supported images or raise a descriptive error."""

    images = list_images(data_dir)
    if not images:
        raise FileNotFoundError(
            f"No supported images found in {data_dir}. "
            f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    return images






def main() -> None:
    args = build_parser().parse_args()
    
    # If --image is "all", process every image in --data_dir and save results to --output_dir.
    if args.image == "all":
        if args.output:
            raise ValueError("--output cannot be used when processing all images; use --output_dir.")
        
        # Raise an error if --data_dir doesn't exist or has no supported images, then initialize
        images = get_images_or_raise(args.data_dir)
        estimator = VLMSizeEstimator(model=args.model)
        total = len(images)
        print(f"Processing {total} images from {args.data_dir}")

        # Process each image and save results, printing progress and summaries to the console.
        for index, image_path in enumerate(images, start=1):
            output_path = default_output_path(image_path, args.output_dir)
            print(f"[{index}/{total}] Processing {image_path.name}...", flush=True)
            payload = estimate_and_write(estimator, image_path, output_path)
            print(
                f"[{index}/{total}] Saved {output_path} "
                f"(height_cm={payload['height_cm']}, width_cm={payload['width_cm']})",
                flush=True,
            )
        return

    # Otherwise, process a single image specified by --image or the first image in --data_dir.
    if args.image:
        image_path = resolve_image_path(args.image, args.data_dir)
    else:
        images = get_images_or_raise(args.data_dir)
        image_path = images[0]

    estimator = VLMSizeEstimator(model=args.model)
    output_path = args.output or default_output_path(image_path, args.output_dir)
    payload = estimate_and_write(estimator, image_path, output_path)
    rendered = json.dumps(payload, indent=2, sort_keys=True)

    print(rendered)


if __name__ == "__main__":
    main()
