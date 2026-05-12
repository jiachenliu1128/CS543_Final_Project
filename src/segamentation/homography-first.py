import cv2
import numpy as np
import torch
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# =========================
# HELPER FUNCTIONS
# =========================
def order_points_robust(pts):
    """Orders 4 coordinates clockwise (TL, TR, BR, BL) using angle sorting."""
    pts = np.array(pts, dtype="float32")
    center_x = np.mean(pts[:, 0])
    center_y = np.mean(pts[:, 1])
    angles = np.arctan2(pts[:, 1] - center_y, pts[:, 0] - center_x)
    sorted_indices = np.argsort(angles)
    return pts[sorted_indices]

def flatten_image_strictly_around_targets(image, corners, target_masks):
    """
    Unwarps the image but dynamically sizes the canvas to strictly fit ONLY 
    the explicitly selected objects.
    """
    rect = order_points_robust(corners)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    side_length = max(int(widthA), int(widthB), int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [side_length - 1, 0],
        [side_length - 1, side_length - 1],
        [0, side_length - 1]
    ], dtype="float32")

    H_matrix = cv2.getPerspectiveTransform(rect, dst)

    safe_points = []
    for m in target_masks:
        x, y, w, h = m["bbox"]
        safe_points.extend([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

    safe_points = np.array(safe_points, dtype="float32").reshape(-1, 1, 2)
    warped_points = cv2.perspectiveTransform(safe_points, H_matrix)

    padding = 100
    x_min, y_min = np.int32(warped_points.min(axis=0).ravel() - padding)
    x_max, y_max = np.int32(warped_points.max(axis=0).ravel() + padding)

    translation_matrix = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ], dtype="float32")

    full_matrix = translation_matrix.dot(H_matrix)
    new_w = x_max - x_min
    new_h = y_max - y_min

    unwarped_image = cv2.warpPerspective(image, full_matrix, (new_w, new_h))
    return unwarped_image

def get_bbox_corners(bbox):
    """
    Converts a standard [x, y, w, h] bounding box into 4 distinct corner coordinates.
    """
    x, y, w, h = map(int, bbox)
    return {
        "TL": (x, y),
        "TR": (x + w, y),
        "BR": (x + w, y + h),
        "BL": (x, y + h)
    }

# =========================
# 1. Config & Setup
# =========================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]

IMAGE_PATH = PROJECT_ROOT / "data" / "543 photos" / "20_BOWL_TILT45_FAR.jpg"
SAM_CHECKPOINT = PROJECT_ROOT / "models" / "sam_vit_b_01ec64.pth"

MODEL_TYPE = "vit_b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)


# =========================
# 2. Load Image & SAM
# =========================
image_bgr = cv2.imread(str(IMAGE_PATH))
if image_bgr is None: raise FileNotFoundError("Cannot read image.")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
H, W = image_rgb.shape[:2]
image_area = H * W

print("\nLoading SAM model...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
sam.to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(
    model=sam, points_per_side=16, pred_iou_thresh=0.86, stability_score_thresh=0.90, crop_n_layers=0
)


# =========================
# 3. SAM Pass 1 (Find Everything)
# =========================
print("Generating initial masks...")
masks = mask_generator.generate(image_rgb)
filtered_masks = [m for m in masks if 0.01 <= (m["area"] / image_area) <= 0.40]
object_masks = sorted(filtered_masks, key=lambda x: x["area"], reverse=True)

display = image_bgr.copy()
colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255)]

for i, m in enumerate(object_masks):
    mask = m["segmentation"]
    color = colors[i % len(colors)]
    layer = np.zeros_like(display)
    layer[mask] = color
    display = cv2.addWeighted(display, 1.0, layer, 0.35, 0)
    
    x, y, w, h = map(int, m["bbox"])
    cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
    center_x, center_y = x + w // 2, y + h // 2
    cv2.putText(display, f"Obj {i + 1}", (center_x - 40, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(display, f"Obj {i + 1}", (center_x - 40, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

scale = min(1.0, 1000 / W)
cv2.imshow("Pass 1: Detect Square and Target", cv2.resize(display, (int(W * scale), int(H * scale))))
print("\nLook at the image window. Remember the object numbers.")
print("Press ANY KEY in the image window to proceed to the terminal prompts.")
cv2.waitKey(0)
cv2.destroyAllWindows()


# =========================
# 4. User Input & Hybrid Selection
# =========================
ref_idx = int(input(f"Enter the Object Number for the WHITE SQUARE [1-{len(object_masks)}]: ")) - 1
target_idx = int(input(f"Enter the Object Number for the TARGET TOOTHPASTE [1-{len(object_masks)}]: ")) - 1

ref_mask_uint8 = object_masks[ref_idx]["segmentation"].astype(np.uint8) * 255
gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
_, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
clean_mask = cv2.bitwise_and(bright_mask, ref_mask_uint8)

contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)
epsilon = 0.04 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)

if len(approx) != 4:
    raise RuntimeError(f"Could not find exactly 4 corners. Found {len(approx)} instead.")

box_points = approx.reshape(-1, 2).astype(np.float32)


# =========================
# 5. Perform Smart-Cropped Homography
# =========================
print("\nFlattening perspective and cropping strictly to targets...")
crucial_objects = [object_masks[ref_idx], object_masks[target_idx]]
unwarped_bgr = flatten_image_strictly_around_targets(image_bgr, box_points, crucial_objects)
unwarped_rgb = cv2.cvtColor(unwarped_bgr, cv2.COLOR_BGR2RGB)
H_flat, W_flat = unwarped_rgb.shape[:2]
flat_image_area = H_flat * W_flat


# =========================
# 6. SAM Pass 2 (On Flat, Cropped Image)
# =========================
print("Running SAM on the perfectly flat, cropped image...")
flat_masks = mask_generator.generate(unwarped_rgb)

flat_filtered = [m for m in flat_masks if 0.005 <= (m["area"] / flat_image_area) <= 0.80]
flat_objects = sorted(flat_filtered, key=lambda x: x["area"], reverse=True)

flat_display = unwarped_bgr.copy()
for i, m in enumerate(flat_objects):
    mask = m["segmentation"]
    color = colors[i % len(colors)]
    layer = np.zeros_like(flat_display)
    layer[mask] = color
    flat_display = cv2.addWeighted(flat_display, 1.0, layer, 0.35, 0)
    
    x, y, w, h = map(int, m["bbox"])
    cv2.rectangle(flat_display, (x, y), (x + w, y + h), color, 2)
    cv2.putText(flat_display, f"Flat Obj {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(flat_display, f"Flat Obj {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

flat_scale = min(1.0, 1000 / max(W_flat, 1))
cv2.imshow("Pass 2: Undistorted & Framed Result", cv2.resize(flat_display, (int(W_flat * flat_scale), int(H_flat * flat_scale))))

print("\nLook at the newly flattened image window.")
print("Press ANY KEY in the image window to proceed to coordinate extraction.")
cv2.waitKey(0)
cv2.destroyAllWindows()


# =========================
# 7. Coordinate Extraction (Pass 2)
# =========================
print("\n" + "="*40)
print("       PASS 2: OBJECT SELECTION")
print("="*40)

# Ask for the Reference Object
print("\n--> Step 1: Select your REFERENCE object (The White Square)")
flat_ref_idx = int(input(f"Enter the Object Number [1-{len(flat_objects)}]: ")) - 1

# Ask for the Target Object
print("\n--> Step 2: Select your TARGET object (The Toothpaste)")
flat_target_idx = int(input(f"Enter the Object Number [1-{len(flat_objects)}]: ")) - 1

# Extract the bounding boxes from SAM's dictionary
ref_bbox = flat_objects[flat_ref_idx]["bbox"]
target_bbox = flat_objects[flat_target_idx]["bbox"]

# Convert to 4 corners
final_square_corners = get_bbox_corners(ref_bbox)
final_target_corners = get_bbox_corners(target_bbox)


# =========================
# 8. Output Final Results
# =========================
print("\n==============================================")
print("             FINAL COORDINATES                ")
print("==============================================")
print("Points are relative to the final unwarped, flat image canvas.\n")

print(f"Reference Square (Flat Obj {flat_ref_idx + 1}) Bounding Box:")
for corner, coords in final_square_corners.items():
    print(f"  {corner}: {coords}")

print(f"\nTarget Toothpaste (Flat Obj {flat_target_idx + 1}) Bounding Box:")
for corner, coords in final_target_corners.items():
    print(f"  {corner}: {coords}")

print("\nPipeline fully complete. These coordinates map exactly to the bounding boxes you see on screen.")