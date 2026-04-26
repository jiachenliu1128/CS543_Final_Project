import cv2
import numpy as np
import torch
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


# =========================
# 1. Config
# =========================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]

IMAGE_PATH = PROJECT_ROOT / "data" / "test.jpg"
SAM_CHECKPOINT = PROJECT_ROOT / "models" / "sam_vit_b_01ec64.pth"

MODEL_TYPE = "vit_b"
DEVICE = "cpu"

print("Torch:", torch.__version__)
print("Device:", DEVICE)
print("Image path:", IMAGE_PATH)
print("Checkpoint path:", SAM_CHECKPOINT)


# =========================
# 2. Load Image
# =========================

image_bgr = cv2.imread(str(IMAGE_PATH))

if image_bgr is None:
    raise FileNotFoundError(f"Cannot read image: {IMAGE_PATH}")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

H, W = image_rgb.shape[:2]
image_area = H * W

print(f"Image size: {W} x {H}")


# =========================
# 3. Load SAM
# =========================

if not SAM_CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Cannot find SAM checkpoint: {SAM_CHECKPOINT}\n"
        f"Download it with:\n"
        f"curl -L -o {SAM_CHECKPOINT} "
        f"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    )

print("Loading SAM model...")

sam = sam_model_registry[MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
sam.to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.90,
    crop_n_layers=0,
)

print("SAM model loaded.")


# =========================
# 4. Generate Masks
# =========================

print("Generating masks...")

masks = mask_generator.generate(image_rgb)

print(f"Generated {len(masks)} masks.")

if len(masks) < 2:
    raise RuntimeError("SAM detected fewer than 2 masks.")


# =========================
# 5. Filter Masks
# =========================

filtered_masks = []

for m in masks:
    area_ratio = m["area"] / image_area

    if area_ratio > 0.40:
        continue

    if area_ratio < 0.01:
        continue

    filtered_masks.append(m)

filtered_masks = sorted(filtered_masks, key=lambda x: x["area"], reverse=True)

print(f"Filtered masks: {len(filtered_masks)}")

if len(filtered_masks) < 2:
    raise RuntimeError("After filtering, fewer than 2 object masks remain.")

object_masks = filtered_masks


# =========================
# 6. Draw All Masks with Object Index
# =========================

display = image_bgr.copy()

colors = [
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (128, 0, 255),
    (255, 128, 0),
    (0, 128, 255),
]

alpha = 0.35

for i, m in enumerate(object_masks):
    mask = m["segmentation"]
    color = colors[i % len(colors)]

    layer = np.zeros_like(display)
    layer[mask] = color
    display = cv2.addWeighted(display, 1.0, layer, alpha, 0)

    x, y, w, h = map(int, m["bbox"])

    cv2.rectangle(
        display,
        (x, y),
        (x + w, y + h),
        color,
        2
    )

    center_x = x + w // 2
    center_y = y + h // 2

    label = f"Object {i + 1}"

    cv2.putText(
        display,
        label,
        (center_x - 60, center_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        4,
        cv2.LINE_AA
    )

    cv2.putText(
        display,
        label,
        (center_x - 60, center_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    print(
        f"Object {i + 1}: "
        f"bbox={m['bbox']}, "
        f"area={m['area']}"
    )


# =========================
# 7. Show Image
# =========================

max_width = 1000
scale = min(1.0, max_width / W)

display_resized = cv2.resize(
    display,
    (int(W * scale), int(H * scale))
)

window_name = "Detected Objects"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.imshow(window_name, display_resized)

print("\nCheck the window and remember object indices.")
print("Press any key in the image window to continue.")

cv2.waitKey(0)
cv2.destroyAllWindows()


# =========================
# 8. Terminal Input
# =========================

num_objects = len(object_masks)

ref_idx = int(input(f"Enter reference object number [1-{num_objects}]: ")) - 1
target_idx = int(input(f"Enter target object number [1-{num_objects}]: ")) - 1

if ref_idx < 0 or ref_idx >= num_objects:
    raise ValueError("Invalid reference object number.")

if target_idx < 0 or target_idx >= num_objects:
    raise ValueError("Invalid target object number.")

if ref_idx == target_idx:
    raise ValueError("Reference and target cannot be the same object.")

length = float(input("Enter reference object real length: "))
width = float(input("Enter reference object real width: "))


# =========================
# 9. Store Result
# =========================

ref_mask = object_masks[ref_idx]["segmentation"]
target_mask = object_masks[target_idx]["segmentation"]

ref_bbox = object_masks[ref_idx]["bbox"]
target_bbox = object_masks[target_idx]["bbox"]

print("\n=== RESULT ===")
print(f"Reference object: Object {ref_idx + 1}")
print(f"Target object: Object {target_idx + 1}")
print(f"Reference bbox: {ref_bbox}")
print(f"Target bbox: {target_bbox}")
print(f"Reference real length: {length}")
print(f"Reference real width: {width}")

print("\nCurrent module finished. Next pipeline is blocked.")