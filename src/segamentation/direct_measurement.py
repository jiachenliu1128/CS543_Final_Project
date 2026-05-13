import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


# =========================
# Helper Functions
# =========================

def order_points_clockwise(pts):
    """
    Order 4 points clockwise around their center.
    This is used only for printing reference corners in a readable order.
    """
    pts = np.array(pts, dtype="float32")
    center_x = np.mean(pts[:, 0])
    center_y = np.mean(pts[:, 1])

    angles = np.arctan2(pts[:, 1] - center_y, pts[:, 0] - center_x)
    sorted_indices = np.argsort(angles)

    return pts[sorted_indices]


def distance(p1, p2):
    """Compute Euclidean distance between two 2D points."""
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def compute_average_side_length(corners):
    """
    Compute average side length from 4 ordered corner points.
    corners should represent the 4 corners of the reference square.
    """
    corners = order_points_clockwise(corners)

    side_lengths = [
        distance(corners[0], corners[1]),
        distance(corners[1], corners[2]),
        distance(corners[2], corners[3]),
        distance(corners[3], corners[0]),
    ]

    return float(np.mean(side_lengths)), corners


def extract_reference_corners(image_bgr, ref_mask):
    """
    Extract 4 reference square corners from the selected reference mask.

    If approxPolyDP does not return exactly 4 corners,
    fallback to minAreaRect to get a rotated rectangle.
    """
    ref_mask_uint8 = ref_mask.astype(np.uint8) * 255
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # The reference marker is white, so threshold bright pixels.
    _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # Keep only the bright area inside the selected reference mask.
    clean_mask = cv2.bitwise_and(bright_mask, ref_mask_uint8)

    # Clean small artifacts.
    kernel = np.ones((3, 3), np.uint8)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        clean_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:
        raise RuntimeError("No contour found for reference square.")

    largest_contour = max(contours, key=cv2.contourArea)

    epsilon = 0.04 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) == 4:
        corners = approx.reshape(-1, 2).astype(np.float32)
        print("Reference corners found using approxPolyDP.")
    else:
        print(f"approxPolyDP found {len(approx)} corners. Using minAreaRect fallback.")
        rect = cv2.minAreaRect(largest_contour)
        corners = cv2.boxPoints(rect).astype(np.float32)

    avg_side_px, ordered_corners = compute_average_side_length(corners)

    return ordered_corners, avg_side_px


def extract_target_rotated_dimensions(target_mask):
    """
    Use minAreaRect on the selected target mask to estimate target length and width in pixels.

    This avoids using a horizontal bounding box.
    """
    target_mask_uint8 = target_mask.astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        target_mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:
        raise RuntimeError("No contour found for target object.")

    largest_contour = max(contours, key=cv2.contourArea)

    rect = cv2.minAreaRect(largest_contour)
    box_points = cv2.boxPoints(rect).astype(np.float32)

    side_a = distance(box_points[0], box_points[1])
    side_b = distance(box_points[1], box_points[2])

    target_length_px = max(side_a, side_b)
    target_width_px = min(side_a, side_b)

    ordered_box_points = order_points_clockwise(box_points)

    return ordered_box_points, target_length_px, target_width_px


def compute_real_world_dimensions(reference_side_px, target_length_px, target_width_px, reference_size_cm):
    """
    Convert target pixel dimensions to real-world cm using the reference square.
    """
    pixels_per_cm = reference_side_px / reference_size_cm

    target_length_cm = target_length_px / pixels_per_cm
    target_width_cm = target_width_px / pixels_per_cm

    return pixels_per_cm, target_length_cm, target_width_cm


def format_points(points):
    """Convert point array to a readable string for CSV."""
    points = np.array(points, dtype=float)
    return "; ".join([f"({p[0]:.1f}, {p[1]:.1f})" for p in points])


def draw_objects(image_bgr, object_masks, window_name):
    """
    Draw SAM masks and object numbers.
    Display is resized to fit the screen.
    """
    display = image_bgr.copy()

    colors = [
        (0, 0, 255),
        (255, 0, 0),
        (0, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 128, 255),
        (128, 0, 255),
    ]

    for i, m in enumerate(object_masks):
        mask = m["segmentation"]
        color = colors[i % len(colors)]

        layer = np.zeros_like(display)
        layer[mask] = color
        display = cv2.addWeighted(display, 1.0, layer, 0.35, 0)

        x, y, w, h = map(int, m["bbox"])
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)

        center_x = x + w // 2
        center_y = y + h // 2

        cv2.putText(
            display,
            f"Obj {i + 1}",
            (center_x - 40, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            4,
        )
        cv2.putText(
            display,
            f"Obj {i + 1}",
            (center_x - 40, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    img_h, img_w = image_bgr.shape[:2]

    max_display_w = 1200
    max_display_h = 800
    scale = min(max_display_w / img_w, max_display_h / img_h, 1.0)

    resized = cv2.resize(display, (int(img_w * scale), int(img_h * scale)))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, resized)

    print("\nLook at the image window and remember the object numbers.")
    print("Press ANY KEY in the image window to continue.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def select_objects_with_confirmation(object_masks):
    """
    Ask the user to select reference and target object numbers.
    User can enter 's' to skip the current image.
    """
    while True:
        print("\nObject selection")
        print("Input order: reference first, target second.")
        print("Enter 's' to skip this image.")

        ref_input = input(f"Enter reference object number [1-{len(object_masks)}] or s: ").strip()

        if ref_input.lower() == "s":
            return None, None, "skipped_by_user"

        target_input = input(f"Enter target object number [1-{len(object_masks)}] or s: ").strip()

        if target_input.lower() == "s":
            return None, None, "skipped_by_user"

        if not ref_input.isdigit() or not target_input.isdigit():
            print("Invalid input. Please enter object numbers only, or s to skip.")
            continue

        ref_idx = int(ref_input) - 1
        target_idx = int(target_input) - 1

        if ref_idx < 0 or ref_idx >= len(object_masks):
            print("Invalid reference object number. Please try again.")
            continue

        if target_idx < 0 or target_idx >= len(object_masks):
            print("Invalid target object number. Please try again.")
            continue

        if ref_idx == target_idx:
            print("Reference and target cannot be the same object. Please try again.")
            continue

        print("\nYou selected:")
        print(f"  Reference = Obj {ref_idx + 1}")
        print(f"  Target    = Obj {target_idx + 1}")

        confirm = input("Confirm this selection? [y/n]: ").strip().lower()

        if confirm == "y":
            return ref_idx, target_idx, "selected"

        print("Selection canceled. Please select again.")


def filter_and_sort_masks(masks, image_area, min_ratio, max_ratio, max_objects=8):
    """
    Filter masks by area ratio and keep only the largest candidates.
    This reduces clutter from background texture and tiny fragments.
    """
    filtered = [
        m for m in masks
        if min_ratio <= (m["area"] / image_area) <= max_ratio
    ]

    sorted_masks = sorted(filtered, key=lambda x: x["area"], reverse=True)

    return sorted_masks[:max_objects]


def print_final_results(
    image_filename,
    reference_corners,
    target_box_points,
    reference_side_px,
    target_length_px,
    target_width_px,
    pixels_per_cm,
    target_length_cm,
    target_width_cm,
):
    """Print final measurement results for checking."""
    print("\n" + "=" * 52)
    print("                  FINAL MEASUREMENT")
    print("=" * 52)
    print(f"Image: {image_filename}")
    print("\nReference square corners:")
    labels = ["P1", "P2", "P3", "P4"]
    for label, point in zip(labels, reference_corners):
        print(f"  {label}: ({point[0]:.1f}, {point[1]:.1f})")

    print("\nTarget rotated rectangle points:")
    for label, point in zip(labels, target_box_points):
        print(f"  {label}: ({point[0]:.1f}, {point[1]:.1f})")

    print("\n" + "=" * 52)
    print("                REAL WORLD DIMENSIONS")
    print("=" * 52)
    print(f"Reference Side: {reference_side_px:.2f} px")
    print(f"Scale Factor:  1 cm = {pixels_per_cm:.2f} pixels")
    print(f"Target Length: {target_length_cm:.2f} cm")
    print(f"Target Width:  {target_width_cm:.2f} cm")
    print(f"Target Length: {target_length_px:.2f} px")
    print(f"Target Width:  {target_width_px:.2f} px")
    print("=" * 52)


def write_failed_row(writer, image_filename, status):
    """Write a failed or skipped row to CSV."""
    writer.writerow(
        {
            "Image Filename": image_filename,
            "Status": status,
        }
    )


def process_one_image(image_path, mask_generator, writer, reference_size_cm=5.0):
    """Process one image and write one row to the output CSV."""
    print("\n" + "=" * 80)
    print(f"Processing image: {image_path.name}")
    print("=" * 80)

    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        print(f"Cannot read image: {image_path}")
        write_failed_row(writer, image_path.name, "failed_read_image")
        return

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = image_rgb.shape[:2]
    image_area = img_h * img_w

    try:
        print("Generating masks...")
        masks = mask_generator.generate(image_rgb)

        object_masks = filter_and_sort_masks(
            masks=masks,
            image_area=image_area,
            min_ratio=0.005,
            max_ratio=0.50,
            max_objects=8,
        )

        print(f"Generated {len(masks)} masks.")
        print(f"Filtered masks shown: {len(object_masks)}")

        if len(object_masks) == 0:
            raise RuntimeError("No valid masks found.")

        for i, m in enumerate(object_masks):
            print(f"Object {i + 1}: bbox={m['bbox']}, area={m['area']}")

        draw_objects(
            image_bgr=image_bgr,
            object_masks=object_masks,
            window_name=f"Direct Measurement: {image_path.name}",
        )

        ref_idx, target_idx, status = select_objects_with_confirmation(object_masks)

        if status == "skipped_by_user":
            print(f"Skipped by user: {image_path.name}")
            write_failed_row(writer, image_path.name, "skipped_by_user")
            return

        reference_corners, reference_side_px = extract_reference_corners(
            image_bgr=image_bgr,
            ref_mask=object_masks[ref_idx]["segmentation"],
        )

        target_box_points, target_length_px, target_width_px = extract_target_rotated_dimensions(
            target_mask=object_masks[target_idx]["segmentation"],
        )

        pixels_per_cm, target_length_cm, target_width_cm = compute_real_world_dimensions(
            reference_side_px=reference_side_px,
            target_length_px=target_length_px,
            target_width_px=target_width_px,
            reference_size_cm=reference_size_cm,
        )

        print_final_results(
            image_filename=image_path.name,
            reference_corners=reference_corners,
            target_box_points=target_box_points,
            reference_side_px=reference_side_px,
            target_length_px=target_length_px,
            target_width_px=target_width_px,
            pixels_per_cm=pixels_per_cm,
            target_length_cm=target_length_cm,
            target_width_cm=target_width_cm,
        )

        writer.writerow(
            {
                "Image Filename": image_path.name,
                "Reference Index": ref_idx + 1,
                "Target Index": target_idx + 1,
                "Reference Side (px)": round(reference_side_px, 4),
                "Reference Size (cm)": reference_size_cm,
                "Scale Factor (px/cm)": round(pixels_per_cm, 4),
                "Target Length (px)": round(target_length_px, 4),
                "Target Width (px)": round(target_width_px, 4),
                "Estimated Target Length (cm)": round(target_length_cm, 4),
                "Estimated Target Width (cm)": round(target_width_cm, 4),
                "Reference Corners": format_points(reference_corners),
                "Target Rotated Rect Points": format_points(target_box_points),
                "Status": "success",
            }
        )

        print(f"Finished: {image_path.name}")

    except Exception as e:
        print(f"Failed on {image_path.name}: {e}")
        write_failed_row(writer, image_path.name, f"failed: {e}")

    finally:
        cv2.destroyAllWindows()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--reference-size-cm", type=float, default=5.0)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parents[1]

    image_dir = project_root / "data" / "543 photos"
    sam_checkpoint = project_root / "models" / "sam_vit_b_01ec64.pth"

    if args.end is None:
        output_name = f"direct_measurement_outputs_{args.start}_end.csv"
    else:
        output_name = f"direct_measurement_outputs_{args.start}_{args.end}.csv"

    output_path = project_root / "results" / output_name

    model_type = "vit_b"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    print("Image directory:", image_dir)
    print("Checkpoint:", sam_checkpoint)
    print("Output:", output_path)
    print(f"Reference size: {args.reference_size_cm} cm x {args.reference_size_cm} cm")

    if not image_dir.exists():
        raise FileNotFoundError(f"Cannot find image directory: {image_dir}")

    if not sam_checkpoint.exists():
        raise FileNotFoundError(f"Cannot find SAM checkpoint: {sam_checkpoint}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.jpeg"))
        + list(image_dir.glob("*.png"))
    )

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {image_dir}")

    total_images = len(image_paths)
    image_paths = image_paths[args.start:args.end]

    print(f"Total images found: {total_images}")
    print(f"Processing images from index {args.start} to {args.end}.")
    print(f"Images in this run: {len(image_paths)}")

    if len(image_paths) == 0:
        raise RuntimeError("No images selected. Check --start and --end values.")

    print("Loading SAM model...")
    sam = sam_model_registry[model_type](checkpoint=str(sam_checkpoint))
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.90,
        crop_n_layers=0,
    )

    fieldnames = [
        "Image Filename",
        "Reference Index",
        "Target Index",
        "Reference Side (px)",
        "Reference Size (cm)",
        "Scale Factor (px/cm)",
        "Target Length (px)",
        "Target Width (px)",
        "Estimated Target Length (cm)",
        "Estimated Target Width (cm)",
        "Reference Corners",
        "Target Rotated Rect Points",
        "Status",
    ]

    with open(output_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for image_path in image_paths:
            process_one_image(
                image_path=image_path,
                mask_generator=mask_generator,
                writer=writer,
                reference_size_cm=args.reference_size_cm,
            )

    print("\nAll selected images processed.")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()