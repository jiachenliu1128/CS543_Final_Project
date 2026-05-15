# CS543 Final Project

This project estimates real-world object dimensions from single RGB images and compares two approaches:

- **Reference-based direct measurement** using a known 5 cm square marker.
- **No-reference VLM baseline** using an OpenAI vision-language model.

The dataset contains 120 images under `data/543 photos/` and metadata/ground truth under `data/cs543_data_collection_plan_updated.csv`. The images vary by object type, object shape, camera pose, and reference placement.

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

For SAM-based reference measurement, download the SAM ViT-B checkpoint:

```bash
mkdir -p models
curl -L -o models/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

For the VLM baseline, create a `.env` file in the repo root:

```bash
OPENAI_API_KEY=...
```

## Reference-Based Direct Measurement

The final reference-based method is implemented in:

```text
src/segamentation/direct_measurement.py
```

This is the method used by the current evaluation CSVs. It does **not** use homography rectification. Instead, it measures directly in the original image:

1. Run SAM automatic mask generation on the image.
2. Filter masks by area and display labeled candidate objects.
3. Manually select the reference square and target object.
4. Extract the reference square from bright pixels inside the selected mask.
5. Estimate the reference square side length in pixels using `approxPolyDP`, with `minAreaRect` fallback.
6. Estimate target length/width in pixels using `cv2.minAreaRect` on the selected target mask.
7. Convert target pixels to centimeters using:

```text
pixels_per_cm = reference_side_px / reference_size_cm
target_length_cm = target_length_px / pixels_per_cm
target_width_cm = target_width_px / pixels_per_cm
```

The longer target side is reported as length and the shorter side as width.

Run a batch by index range:

```bash
python src/segamentation/direct_measurement.py --start 0 --end 40
python src/segamentation/direct_measurement.py --start 40 --end 80
python src/segamentation/direct_measurement.py --start 80 --end 120
```

Outputs are written to:

```text
results/direct_measurement_outputs_<start>_<end>.csv
```

The currently expected reference-result files are:

```text
results/direct_measurement_outputs_0_40.csv
results/direct_measurement_outputs_40_80.csv
results/direct_measurement_outputs_80_120.csv
```

`src/segamentation/homography-first.py` is an exploratory earlier approach that rectified the image using a perspective transform. It is not the final method used in the current evaluation.

## VLM Baseline

The no-reference VLM baseline is implemented in:

```text
src/vlm_baseline/
scripts/run_vlm_baseline.py
```

It sends each image to an OpenAI vision model with a strict JSON schema and asks for the target object's top-down footprint dimensions. The prompt explicitly tells the model to ignore the white square reference marker, so the baseline does not use geometric calibration.

Run one image:

```bash
python scripts/run_vlm_baseline.py 01_DRIVERSLIC_TD_NEAR.jpg
```

Run all images:

```bash
python scripts/run_vlm_baseline.py all
```

Default image directory:

```text
data/543 photos
```

Default output directory:

```text
results/vlm_baseline/
```

Each output JSON has:

```json
{
  "height_cm": 14.0,
  "width_cm": 7.2
}
```

The default model is `gpt-5.4-mini`. Override it with `--model` or `VLM_BASELINE_MODEL`.

## Evaluation

The evaluation notebook is:

```text
notebooks/evaluation_reference_vs_vlm.ipynb
```

It loads:

- Ground truth: `data/cs543_data_collection_plan_updated.csv`
- VLM baseline JSONs: `results/vlm_baseline/*_vlm_baseline.json`
- Reference-based CSVs: `results/direct_measurement_outputs_*.csv`

The notebook concatenates the split reference CSVs vertically, derives `Photo ID` from `Image Filename`, and evaluates only photos with both baseline and reference estimates.

Before computing errors, all dimensions are orientation-normalized:

- longer side = length
- shorter side = width

This avoids penalizing either method for swapping length and width axes.

Metrics and analyses include:

- MAE and MAPE for baseline and reference-based estimates.
- Paired reference-vs-baseline deltas.
- Paired t-test and Wilcoxon signed-rank test.
- Bootstrap confidence intervals.
- Category-level analysis by known/unknown, regular/irregular, camera pose, reference near/far, object group, and interactions.
- Signed bias analysis by camera pose, including the observed tendency for tilted images to underestimate and rotated images to overestimate.

Evaluation tables are exported to:

```text
results/evaluation/
```

## Notes

- The final reference-based method assumes the reference marker and target are measured in the same image plane. It does not correct perspective foreshortening.
- Because of that, tilted-camera images can show systematic bias.
- The SAM scripts use OpenCV GUI windows and require local display support.
