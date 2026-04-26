import os
import glob
from collections import Counter
from ultralytics import RTDETR
import cv2
import numpy as np

# --- CONFIGURATION ---
MODEL_PATH = "yolo_dataset_augmented/trainresults/run_full/weights/best.pt"
IMAGES_DIR = "yolo_dataset_augmented/images/train"
LABELS_DIR = "yolo_dataset_augmented//labels/train"
OUTPUT_DIR = "failures_analysisv2"
CONF_THRESHOLD = 0.76
CLASS_NAMES = ["table", "image_large_symbol", "image_icon", "text_content"]

CLASS_COLORS = {
    "table": (255, 0, 0),             # Blue
    "image_large_symbol": (0, 255, 0),  # Green
    "image_icon": (0, 0, 255),          # Red
    "text_content": (255, 255, 0)       # Cyan
}

def read_yolo_labels(label_path):
    """Reads ground truth labels with bounding boxes."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_idx = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                if 0 <= cls_idx < len(CLASS_NAMES):
                    boxes.append({
                        'class': CLASS_NAMES[cls_idx],
                        'cx': cx, 'cy': cy, 'w': w, 'h': h,
                        'conf': None # GT has no confidence
                    })
    return boxes

def count_boxes(boxes):
    """Count boxes by class."""
    counts = Counter()
    for box in boxes:
        counts[box['class']] += 1
    return counts

def draw_boxes(img, boxes, color_map, label_prefix=""):
    """Draw bounding boxes on image with confidence."""
    h, w = img.shape[:2]
    img_copy = img.copy()

    for box in boxes:
        cls_name = box['class']
        color = color_map.get(cls_name, (255, 255, 255))
        confidence = box.get('conf')

        cx = int(box['cx'] * w)
        cy = int(box['cy'] * h)
        bw = int(box['w'] * w)
        bh = int(box['h'] * h)

        x1 = cx - bw // 2
        y1 = cy - bh // 2
        x2 = cx + bw // 2
        y2 = cy + bh // 2

        # Draw Rectangle
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

        # Build Label String
        label = f"{label_prefix}{cls_name}"
        if confidence is not None:
            label += f" {confidence:.2f}"

        # Draw Label Background & Text
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_copy, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
        cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img_copy

def get_model_preds(model, img_path):
    """Gets model predictions with bounding boxes and confidence scores."""
    # Run inference
    results = model.predict(source=img_path, conf=CONF_THRESHOLD, verbose=False)[0]
    boxes = []

    if results.boxes and len(results.boxes) > 0:
        h, w = results.orig_shape

        for i in range(len(results.boxes)):
            cls_idx = int(results.boxes.cls[i].item())
            conf_score = float(results.boxes.conf[i].item()) # Get confidence score

            if 0 <= cls_idx < len(CLASS_NAMES):
                x1, y1, x2, y2 = results.boxes.xyxy[i].cpu().numpy()

                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                boxes.append({
                    'class': CLASS_NAMES[cls_idx],
                    'cx': cx, 'cy': cy, 'w': bw, 'h': bh,
                    'conf': conf_score  # Store confidence
                })

    return boxes

def main():
    # 1. Create main output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 2. Create subdirectories for each class
    for cls_name in CLASS_NAMES:
        class_dir = os.path.join(OUTPUT_DIR, cls_name)
        os.makedirs(class_dir, exist_ok=True)
        print(f"Created directory: {class_dir}")

    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = RTDETR(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Analyzing images in {IMAGES_DIR}...")
    img_files = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.*")))

    failures = []
    total_images = 0

    report_path = os.path.join(OUTPUT_DIR, "failure_report.txt")

    with open(report_path, "w") as f_out:
        f_out.write(f"FAILURE REPORT WITH VISUALIZATIONS\n")
        f_out.write(f"Conf Threshold: {CONF_THRESHOLD}\n")
        f_out.write("="*60 + "\n\n")

        for img_path in img_files:
            total_images += 1
            filename = os.path.basename(img_path)
            label_file = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(LABELS_DIR, label_file)

            # Get GT and Pred boxes
            gt_boxes = read_yolo_labels(label_path)
            pred_boxes = get_model_preds(model, img_path)

            # Count by class
            gt_counts = count_boxes(gt_boxes)
            pred_counts = count_boxes(pred_boxes)

            # Check for mismatches
            all_classes = set(gt_counts.keys()) | set(pred_counts.keys())
            
            # Track which specific classes failed for this image
            failed_classes_in_image = []
            img_errors = []

            for cls in all_classes:
                gt = gt_counts[cls]
                pred = pred_counts[cls]

                if gt != pred:
                    failed_classes_in_image.append(cls)
                    diff = pred - gt
                    if diff < 0:
                        msg = f"MISSING {abs(diff)} {cls} (FN)"
                    else:
                        msg = f"EXTRA {diff} {cls} (FP)"
                    img_errors.append(msg)

            # If there are errors, save visualization to the specific class folders
            if failed_classes_in_image:
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    continue

                # Create GT and Pred visualizations
                gt_img = draw_boxes(img, gt_boxes, CLASS_COLORS, label_prefix="GT:")
                pred_img = draw_boxes(img, pred_boxes, CLASS_COLORS, label_prefix="Pred:")

                # Create side-by-side comparison
                h, w = img.shape[:2]
                label_height = 40
                
                gt_label = np.zeros((label_height, w, 3), dtype=np.uint8)
                pred_label = np.zeros((label_height, w, 3), dtype=np.uint8)

                cv2.putText(gt_label, "GROUND TRUTH", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(pred_label, "PREDICTION", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                gt_with_label = np.vstack([gt_label, gt_img])
                pred_with_label = np.vstack([pred_label, pred_img])
                comparison = np.hstack([gt_with_label, pred_with_label])

                # Save comparison image to EACH failed class folder
                output_filename = f"{os.path.splitext(filename)[0]}.jpg"
                saved_locations = []
                
                for cls_name in failed_classes_in_image:
                    if cls_name in CLASS_NAMES: # Safety check
                        save_path = os.path.join(OUTPUT_DIR, cls_name, output_filename)
                        cv2.imwrite(save_path, comparison)
                        saved_locations.append(cls_name)

                # Write to report
                error_summary = ", ".join(img_errors)
                log_entry = f"[FAILED] {filename}\n"
                log_entry += f"  Ground Truth: {dict(gt_counts)}\n"
                log_entry += f"  Prediction:   {dict(pred_counts)}\n"
                log_entry += f"  Issue:        {error_summary}\n"
                log_entry += f"  Saved to:     {saved_locations}\n"
                log_entry += "-"*60 + "\n"

                print(log_entry.strip())
                f_out.write(log_entry)
                failures.append(filename)

        summary = f"\nAnalysis Complete.\nTotal Images: {total_images}\nFailed Images: {len(failures)}\nReport saved to: {report_path}\n"
        print(summary)
        f_out.write(summary)

if __name__ == "__main__":
    main()
