"""OpenAI VLM baseline for direct object-size estimation from an image."""

from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import OpenAI

from .prompts import SYSTEM_PROMPT, USER_PROMPT

DEFAULT_MODEL = "gpt-5.4-mini"

SIZE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "height_cm": {
            "type": "number",
            "description": "Estimated front-to-back top-down object dimension in centimeters.",
        },
        "width_cm": {
            "type": "number",
            "description": "Estimated left-to-right top-down object dimension in centimeters.",
        },
    },
    "required": ["height_cm", "width_cm"],
}






@dataclass(frozen=True)
class SizeEstimate:
    """Structured physical size estimate in centimeters."""

    height_cm: float
    width_cm: float

    def to_dict(self) -> dict[str, float]:
        return {"height_cm": self.height_cm, "width_cm": self.width_cm}


def image_to_data_url(image_path: Path) -> str:
    """Encode a local image as a data URL accepted by vision APIs."""

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"Expected a file path, got: {image_path}")

    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None or not mime_type.startswith("image/"):
        raise ValueError(f"Unsupported image type for: {image_path}")

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def parse_size_estimate(raw_text: str) -> SizeEstimate:
    """Parse and validate the model's JSON response."""

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned invalid JSON: {raw_text}") from exc

    try:
        height_cm = float(payload["height_cm"])
        width_cm = float(payload["width_cm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Model JSON is missing numeric height_cm/width_cm: {payload}") from exc

    if height_cm <= 0 or width_cm <= 0:
        raise ValueError(f"Model returned non-positive dimensions: {payload}")

    return SizeEstimate(height_cm=height_cm, width_cm=width_cm)








class VLMSizeEstimator:
    """Direct no-reference VLM size-estimation baseline."""

    def __init__(self, model: str = DEFAULT_MODEL, client: "OpenAI | None" = None) -> None:
        self.model = model
        if client is not None:
            self.client = client
            return

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The OpenAI SDK is required for VLM inference. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        from dotenv import load_dotenv

        load_dotenv()
        self.client = OpenAI()

    def estimate(self, image_path: str | Path) -> SizeEstimate:
        """Estimate the centered object's top-down height and width in centimeters."""

        data_url = image_to_data_url(Path(image_path))
        response = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": USER_PROMPT},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "object_size_estimate",
                    "strict": True,
                    "schema": SIZE_SCHEMA,
                }
            },
        )
        return parse_size_estimate(response.output_text)





def estimate_image_size(image_path: str | Path, model: str = DEFAULT_MODEL) -> dict[str, float]:
    """Convenience function for one-off baseline inference."""

    return VLMSizeEstimator(model=model).estimate(image_path).to_dict()
