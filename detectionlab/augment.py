import os
import json
import random
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ruamel.yaml import YAML

# ================= CONFIGURATION =================
INPUT_ROOT = Path("yolo_dataset")
OUTPUT_ROOT = Path(f"{INPUT_ROOT.name}_augmented")

IN_IMGS = INPUT_ROOT / "images" / "train"
IN_LBLS = INPUT_ROOT / "labels" / "train"

OUT_IMGS = OUTPUT_ROOT / "images" / "train"
OUT_LBLS = OUTPUT_ROOT / "labels" / "train"

CLASS_MAPPING = {
    0: "table",
    1: "image_icon",
    2: "large_image_icon",
}

OVERLAP_CLASSES = [1, 2]
TARGET_INSTANCES = {
    0: 20000,
    1: 20000,
    2: 20000,
}

TARGET_EMPTY_TABLES = 1000
MAX_OBJECTS_PER_SHEET = 14

PAPER_SIZES = [
    (1754, 1240),
    (2480, 1754),
    (3508, 2480),
    (1240, 1754),
    (1754, 2480),
    (2480, 3508),
]

NOISE_CLASS_NAMES = {"text_block", "text_bloc", "text_content"}
NOISE_PER_SHEET_PROB = 0.6
MAX_NOISE_PER_SHEET = 3

OVERLAP_MODE_WEIGHTS = {"overlap": 1/3, "border": 1/3, "normal": 1/3}
SMALL_UNDER_LARGE_PROB = 0.65
BORDER_ZONE = 0.05

FRAME_CANVAS_PROB = 1 / 3
OUTSIDE_FRAME_PROB = 0.90
PAIRED_SNAP_PROB = 0.90
TABLE_MARGIN_WEIGHTS = [0.35, 0.10, 0.10, 0.45]
DRAWING_FRAME_DIR = Path("drawing_frame")

TABLE_MARGIN_MAX_SCALE = 1
TABLE_MARGIN_MIN_SCALE = 0.50

# Failure images to be copied 5x into augmented dataset
FAILURE_IMAGE_STEMS = [
    "v#2_BACKPLANE_B22.10002.001E_1",
    "v2_BACKPLANE_B22.10002.001E_2",
    "v2_BACKPLANE_B22.10002.001E_3",
    "v2_BACKPLANE_B22.10002.001M_1",
    "v2_BACKPLANE_B22.10002.001M_2",
    "v1_batch1_B62_5",
    "v1_batch1_B62_6",
]
FAILURE_COPY_COUNT = 5
# =================================================


# --------------------------------------------------
# SETUP
# --------------------------------------------------
def setup_new_dataset():
    if OUTPUT_ROOT.exists():
        print(f"Warning: Output folder '{OUTPUT_ROOT}' already exists.")
    print(f"Creating new dataset at: {OUTPUT_ROOT}")
    OUT_IMGS.mkdir(parents=True, exist_ok=True)
    OUT_LBLS.mkdir(parents=True, exist_ok=True)

    failure_stems_set = set(FAILURE_IMAGE_STEMS)

    print("Copying original dataset (1x) + failure images (5x)...")
    for img_path in tqdm(list(IN_IMGS.glob("*")), desc="Copying Originals"):
        if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue
        src_lbl = IN_LBLS / f"{img_path.stem}.txt"

        if img_path.stem in failure_stems_set:
            for copy_idx in range(FAILURE_COPY_COUNT):
                suffix = "" if copy_idx == 0 else f"_fail{copy_idx}"
                dst_name = f"{img_path.stem}{suffix}{img_path.suffix}"
                shutil.copy(img_path, OUT_IMGS / dst_name)
                if src_lbl.exists():
                    shutil.copy(src_lbl, OUT_LBLS / f"{img_path.stem}{suffix}.txt")
        else:
            dst_name = f"{img_path.stem}{img_path.suffix}"
            shutil.copy(img_path, OUT_IMGS / dst_name)
            if src_lbl.exists():
                shutil.copy(src_lbl, OUT_LBLS / f"{img_path.stem}.txt")

    yaml_src = INPUT_ROOT / "data.yaml"
    if yaml_src.exists():
        shutil.copy(yaml_src, OUTPUT_ROOT / "data.yaml")
        ryaml = YAML()
        ryaml.preserve_quotes = True
        yaml_dst = OUTPUT_ROOT / "data.yaml"
        with open(yaml_dst, encoding="utf-8") as f:
            data = ryaml.load(f)
        data["train"] = str(OUTPUT_ROOT / "images" / "train")
        data.pop("val", None)
        with open(yaml_dst, "w", encoding="utf-8") as f:
            ryaml.dump(data, f)


# --------------------------------------------------
# SAVE SAMPLE
# --------------------------------------------------
def save_new_sample(img, boxes, prefix, idx):
    filename = f"{prefix}_{idx}"
    cv2.imwrite(str(OUT_IMGS / f"{filename}.png"), img)
    with open(OUT_LBLS / f"{filename}.txt", "w") as f:
        for (cls, xc, yc, w, h) in boxes:
            xc = float(np.clip(xc, 0.0, 1.0))
            yc = float(np.clip(yc, 0.0, 1.0))
            w  = float(np.clip(w,  0.0, 1.0))
            h  = float(np.clip(h,  0.0, 1.0))
            if w <= 0 or h <= 0:
                continue
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


# --------------------------------------------------
# CANVAS SIZE
# --------------------------------------------------
def get_random_canvas_size():
    if random.random() < (15.0 / 16.0):
        return random.choice([(1754, 1240), (2480, 1754), (3508, 2480)])
    else:
        return random.choice([(1240, 1754), (1754, 2480), (2480, 3508)])


# --------------------------------------------------
# COLLISION CHECK
# --------------------------------------------------
def check_collision(new_box, new_cls, occupied_list, overlap_classes, max_iou=0.20):
    nx1, ny1, nx2, ny2 = new_box
    area_new = (nx2 - nx1) * (ny2 - ny1)
    if area_new <= 0:
        return True
    for (ex1, ey1, ex2, ey2, e_cls) in occupied_list:
        ix1 = max(nx1, ex1); iy1 = max(ny1, ey1)
        ix2 = min(nx2, ex2); iy2 = min(ny2, ey2)
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if intersection > 0:
            if new_cls not in overlap_classes or e_cls not in overlap_classes:
                return True
            area_ex = (ex2 - ex1) * (ey2 - ey1)
            union = area_new + area_ex - intersection
            if union > 0 and (intersection / union) > max_iou:
                return True
    return False


# --------------------------------------------------
# PLACEMENT HELPERS
# --------------------------------------------------
def _weighted_choice(weights_dict):
    keys = list(weights_dict.keys())
    vals = list(weights_dict.values())
    return random.choices(keys, weights=vals, k=1)[0]

def _ri(a, b):
    a, b = int(a), int(b)
    if a > b:
        a = b
    return random.randint(a, b)

def place_border(W, H, w_obj, h_obj, x_min=0, y_min=0, x_max=None, y_max=None):
    if x_max is None: x_max = W
    if y_max is None: y_max = H
    bx = int(W * BORDER_ZONE)
    by = int(H * BORDER_ZONE)
    x_lo = x_min;  x_hi = max(x_min, x_max - w_obj)
    y_lo = y_min;  y_hi = max(y_min, y_max - h_obj)
    edge = random.choice(["left", "right", "top", "bottom"])
    if edge == "left":
        x = _ri(x_lo, min(x_lo + bx, x_hi)); y = _ri(y_lo, y_hi)
    elif edge == "right":
        x = _ri(max(x_lo, x_hi - bx), x_hi); y = _ri(y_lo, y_hi)
    elif edge == "top":
        x = _ri(x_lo, x_hi); y = _ri(y_lo, min(y_lo + by, y_hi))
    else:
        x = _ri(x_lo, x_hi); y = _ri(max(y_lo, y_hi - by), y_hi)
    return int(np.clip(x, x_lo, x_hi)), int(np.clip(y, y_lo, y_hi))

def place_overlapping(W, H, w_obj, h_obj, occupied,
                      x_min=0, y_min=0, x_max=None, y_max=None):
    if x_max is None: x_max = W
    if y_max is None: y_max = H
    x_lo = x_min; x_hi = max(x_min, x_max - w_obj)
    y_lo = y_min; y_hi = max(y_min, y_max - h_obj)
    anchors = [(ex1, ey1, ex2, ey2)
               for (ex1, ey1, ex2, ey2, ec) in occupied if ec in OVERLAP_CLASSES]
    if not anchors:
        return _ri(x_lo, x_hi), _ri(y_lo, y_hi)
    ex1, ey1, ex2, ey2 = random.choice(anchors)
    sx = int(random.uniform(0.10, 0.40) * min(w_obj, ex2 - ex1))
    sy = int(random.uniform(0.10, 0.40) * min(h_obj, ey2 - ey1))
    if random.random() < 0.5: sx = -sx
    if random.random() < 0.5: sy = -sy
    return int(np.clip(ex1 + sx, x_lo, x_hi)), int(np.clip(ey1 + sy, y_lo, y_hi))

def place_small_under_large(W, H, w_small, h_small, occupied,
                             x_min=0, y_min=0, x_max=None, y_max=None):
    if x_max is None: x_max = W
    if y_max is None: y_max = H
    x_lo = x_min; x_hi = max(x_min, x_max - w_small)
    y_lo = y_min; y_hi = max(y_min, y_max - h_small)
    if random.random() > SMALL_UNDER_LARGE_PROB:
        return _ri(x_lo, x_hi), _ri(y_lo, y_hi)
    large_anchors = sorted(
        [(ex1, ey1, ex2, ey2)
         for (ex1, ey1, ex2, ey2, ec) in occupied if ec in OVERLAP_CLASSES],
        key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
        reverse=True,
    )
    if not large_anchors:
        return _ri(x_lo, x_hi), _ri(y_lo, y_hi)
    ex1, ey1, ex2, ey2 = large_anchors[0]
    mid_y = (ey1 + ey2) // 2
    x = int(np.clip(_ri(ex1, max(ex1, ex2 - w_small)), x_lo, x_hi))
    y = int(np.clip(_ri(mid_y, max(mid_y, ey2 - h_small // 2)), y_lo, y_hi))
    return x, y


# --------------------------------------------------
# MARGIN PLACEMENT
# --------------------------------------------------
def _margin_strip_bounds(zone, W, H, inner_rect):
    ix, iy   = inner_rect["x"],  inner_rect["y"]
    ix2, iy2 = inner_rect["x2"], inner_rect["y2"]
    if zone == "top":      return 0,   0,   W,   iy
    elif zone == "bottom": return 0,   iy2, W,   H
    elif zone == "left":   return 0,   0,   ix,  H
    else:                  return ix2, 0,   W,   H

def _resize_to_fit(obj_img, internal_boxes, max_w, max_h):
    oh, ow = obj_img.shape[:2]
    if max_w <= 0 or max_h <= 0:
        return None, None, None
    scale = min(max_w / ow, max_h / oh)
    scale = min(scale, TABLE_MARGIN_MAX_SCALE)
    if scale < TABLE_MARGIN_MIN_SCALE:
        return None, None, None
    if abs(scale - 1.0) < 0.02:
        return obj_img, internal_boxes, 1.0
    nw = max(1, int(ow * scale))
    nh = max(1, int(oh * scale))
    resized = cv2.resize(obj_img, (nw, nh),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    new_boxes = [(c,
                  int(bx1 * scale), int(by1 * scale),
                  int(bx2 * scale), int(by2 * scale))
                 for (c, bx1, by1, bx2, by2) in internal_boxes]
    return resized, new_boxes, scale

def place_table_in_margin(W, H, obj_img, internal_boxes, inner_rect):
    ix, iy   = inner_rect["x"],  inner_rect["y"]
    ix2, iy2 = inner_rect["x2"], inner_rect["y2"]
    zones = ["top", "bottom", "left", "right"]
    random.shuffle(zones)
    zones = random.choices(zones, weights=TABLE_MARGIN_WEIGHTS, k=len(zones))
    seen = set(); ordered_zones = []
    for z in zones:
        if z not in seen:
            seen.add(z); ordered_zones.append(z)
    for z in ["top", "bottom", "left", "right"]:
        if z not in seen:
            ordered_zones.append(z)
    for zone in ordered_zones:
        sx_lo, sy_lo, sx_hi, sy_hi = _margin_strip_bounds(zone, W, H, inner_rect)
        strip_w = sx_hi - sx_lo
        strip_h = sy_hi - sy_lo
        fit_img, fit_boxes, _ = _resize_to_fit(obj_img, internal_boxes, strip_w, strip_h)
        if fit_img is None:
            continue
        fh, fw = fit_img.shape[:2]
        x_lo_p = sx_lo;  x_hi_p = sx_hi - fw
        y_lo_p = sy_lo;  y_hi_p = sy_hi - fh
        if x_hi_p < x_lo_p or y_hi_p < y_lo_p:
            continue
        STICK_JITTER = 5
        if zone == "top":
            y = int(np.clip(iy - fh + _ri(-STICK_JITTER, STICK_JITTER), y_lo_p, y_hi_p))
            x = _ri(x_lo_p, x_hi_p)
        elif zone == "bottom":
            y = int(np.clip(iy2 + _ri(-STICK_JITTER, STICK_JITTER), y_lo_p, y_hi_p))
            x = _ri(x_lo_p, x_hi_p)
        elif zone == "left":
            x = int(np.clip(ix - fw + _ri(-STICK_JITTER, STICK_JITTER), x_lo_p, x_hi_p))
            y = _ri(y_lo_p, y_hi_p)
        else:
            x = int(np.clip(ix2 + _ri(-STICK_JITTER, STICK_JITTER), x_lo_p, x_hi_p))
            y = _ri(y_lo_p, y_hi_p)
        x = int(np.clip(x, x_lo_p, x_hi_p))
        y = int(np.clip(y, y_lo_p, y_hi_p))
        return fit_img, fit_boxes, x, y
    return None


# --------------------------------------------------
# DRAWING FRAME LOADER
# --------------------------------------------------
def load_frame_assets():
    json_path = DRAWING_FRAME_DIR / "frame_edges.json"
    if not json_path.exists():
        print(f"Warning: {json_path} not found — frame canvas disabled.")
        return None, None
    with open(json_path, encoding="utf-8") as f:
        frame_meta = json.load(f)
    frame_images = {}
    for orient in ("horizontal", "vertical"):
        frame_images[orient] = {}
        if orient not in frame_meta:
            continue
        for key in frame_meta[orient]:
            img_path = DRAWING_FRAME_DIR / orient / f"{key}.png"
            if not img_path.exists():
                print(f"Warning: frame image not found: {img_path}")
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: could not read: {img_path}")
                continue
            frame_images[orient][key] = img
    return frame_meta, frame_images

def pick_random_frame(frame_meta, frame_images):
    available = [o for o in ("horizontal", "vertical")
                 if o in frame_meta and frame_images.get(o)]
    if not available:
        return None
    _orient_weights = [1 if o == "vertical" else 15 for o in available]
    orient = random.choices(available, weights=_orient_weights, k=1)[0]
    valid_keys = list(frame_images[orient].keys())
    if not valid_keys:
        return None
    key = random.choice(valid_keys)
    meta = frame_meta[orient][key]
    W, H = meta["width"], meta["height"]
    inner = meta["inner_rect"]
    canvas = frame_images[orient][key].copy()
    return canvas, W, H, inner


# --------------------------------------------------
# SYNTHETIC EMPTY TABLE
# --------------------------------------------------
def generate_synthetic_empty_table():
    n_cols = random.randint(2, 4)
    n_rows = random.randint(2, 4)
    col_widths  = [random.randint(60, 280) for _ in range(n_cols)]
    row_heights = [random.randint(30, 120) for _ in range(n_rows)]
    total_w = sum(col_widths)
    total_h = sum(row_heights)
    pad = 1
    canvas_w = total_w + pad * 2
    canvas_h = total_h + pad * 2
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    line_color = (38, 38, 38)
    cv2.rectangle(canvas, (pad, pad), (pad + total_w, pad + total_h), line_color, 1)
    x_cursor = pad
    for cw in col_widths[:-1]:
        x_cursor += cw
        cv2.line(canvas, (x_cursor, pad), (x_cursor, pad + total_h), line_color, 1)
    y_cursor = pad
    for rh in row_heights[:-1]:
        y_cursor += rh
        cv2.line(canvas, (pad, y_cursor), (pad + total_w, y_cursor), line_color, 1)
    internal_boxes = [(0, 0, 0, canvas_w, canvas_h)]
    return canvas, internal_boxes


# --------------------------------------------------
# FAKE CONNECTED TABLE
# --------------------------------------------------
def create_fake_connected_table(crops_class_0, max_dim=1400):
    if len(crops_class_0) < 2:
        return None
    img1, boxes1 = random.choice(crops_class_0)
    img2, boxes2 = random.choice(crops_class_0)
    for _ in range(5):
        if img1.shape != img2.shape:
            break
        img2, boxes2 = random.choice(crops_class_0)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    side = random.choice(["right", "left", "below", "above"])
    if side in ("right", "left"):
        align = random.choice(["top", "bottom", "center"])
        h_c = max(h1, h2); w_c = w1 + w2
        if align == "top":      y1_off, y2_off = 0, 0
        elif align == "bottom": y1_off, y2_off = h_c - h1, h_c - h2
        else:                   y1_off, y2_off = (h_c - h1) // 2, (h_c - h2) // 2
        canvas = np.full((h_c, w_c, 3), 255, dtype=np.uint8)
        if side == "right":
            canvas[y1_off:y1_off+h1, 0:w1] = img1
            canvas[y2_off:y2_off+h2, w1:w1+w2] = img2
            new_boxes  = [(c, x1, y1+y1_off, x2, y2+y1_off) for (c,x1,y1,x2,y2) in boxes1]
            new_boxes += [(c, x1+w1, y1+y2_off, x2+w1, y2+y2_off) for (c,x1,y1,x2,y2) in boxes2]
        else:
            canvas[y2_off:y2_off+h2, 0:w2] = img2
            canvas[y1_off:y1_off+h1, w2:w2+w1] = img1
            new_boxes  = [(c, x1+w2, y1+y1_off, x2+w2, y2+y1_off) for (c,x1,y1,x2,y2) in boxes1]
            new_boxes += [(c, x1, y1+y2_off, x2, y2+y2_off) for (c,x1,y1,x2,y2) in boxes2]
    else:
        align = random.choice(["left", "right", "center"])
        w_c = max(w1, w2); h_c = h1 + h2
        if align == "left":    x1_off, x2_off = 0, 0
        elif align == "right": x1_off, x2_off = w_c - w1, w_c - w2
        else:                  x1_off, x2_off = (w_c - w1) // 2, (w_c - w2) // 2
        canvas = np.full((h_c, w_c, 3), 255, dtype=np.uint8)
        if side == "below":
            canvas[0:h1, x1_off:x1_off+w1] = img1
            canvas[h1:h1+h2, x2_off:x2_off+w2] = img2
            new_boxes  = [(c, x1+x1_off, y1, x2+x1_off, y2) for (c,x1,y1,x2,y2) in boxes1]
            new_boxes += [(c, x1+x2_off, y1+h1, x2+x2_off, y2+h1) for (c,x1,y1,x2,y2) in boxes2]
        else:
            canvas[0:h2, x2_off:x2_off+w2] = img2
            canvas[h2:h2+h1, x1_off:x1_off+w1] = img1
            new_boxes  = [(c, x1+x1_off, y1+h2, x2+x1_off, y2+h2) for (c,x1,y1,x2,y2) in boxes1]
            new_boxes += [(c, x1+x2_off, y1, x2+x2_off, y2) for (c,x1,y1,x2,y2) in boxes2]
    h_c, w_c = canvas.shape[:2]
    scale = min(max_dim / w_c, max_dim / h_c, 1.0)
    if scale < 1.0:
        new_w = max(1, int(w_c * scale))
        new_h = max(1, int(h_c * scale))
        canvas = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_AREA)
        new_boxes = [(c, int(x1*scale), int(y1*scale), int(x2*scale), int(y2*scale))
                     for c, x1, y1, x2, y2 in new_boxes]
    return canvas, new_boxes


# --------------------------------------------------
# PASTE NOISE
# --------------------------------------------------
def paste_noise_elements(canvas, occupied, noise_crops, W, H):
    if not noise_crops:
        return
    n = random.randint(1, MAX_NOISE_PER_SHEET)
    for _ in range(n):
        crop, _ = random.choice(noise_crops)
        h_n, w_n = crop.shape[:2]
        if h_n >= H or w_n >= W:
            continue
        for _ in range(30):
            x = random.randint(0, W - w_n)
            y = random.randint(0, H - h_n)
            if not check_collision((x, y, x+w_n, y+h_n), -1, occupied, overlap_classes=[]):
                canvas[y:y+h_n, x:x+w_n] = crop
                occupied.append((x, y, x+w_n, y+h_n, -1))
                break


# --------------------------------------------------
# LOAD CROPS
# --------------------------------------------------
def load_crops_from_original():
    print("Extracting crops from original data...")
    crops_by_class = {k: [] for k in CLASS_MAPPING.keys()}
    connected_tables = []
    connected_tables_full = []
    noise_crops = []
    orig_counts = {k: 0 for k in CLASS_MAPPING.keys()}

    img_files = sorted([f for f in IN_IMGS.iterdir()
                        if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])

    noise_ids = set()
    yaml_path = INPUT_ROOT / "data.yaml"
    if yaml_path.exists():
        ryaml = YAML()
        with open(yaml_path, encoding="utf-8") as f:
            ydata = ryaml.load(f)
        names = ydata.get("names", [])
        # If names is a dict (e.g., {0: "table", 1: "textblock"}), extract values
        if isinstance(names, dict):
            names = list(names.values())

        if isinstance(names, dict):
            names = list(names.values())
        for i, name in enumerate(names):
            if str(name).lower() in {n.lower() for n in NOISE_CLASS_NAMES}:
                noise_ids.add(i)
        print(f"Noise class IDs detected: {noise_ids}")

    for img_path in tqdm(img_files, desc="Harvesting Crops"):
        label_path = IN_LBLS / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]
        with open(label_path, "r") as f:
            lines = f.readlines()
        boxes_abs = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            x1 = max(0, int((cx - w/2) * w_img))
            y1 = max(0, int((cy - h/2) * h_img))
            x2 = min(w_img, int((cx + w/2) * w_img))
            y2 = min(h_img, int((cy + h/2) * h_img))
            if x2 <= x1 or y2 <= y1:
                continue
            if cls_id in noise_ids:
                noise_crops.append((img[y1:y2, x1:x2], []))
            elif cls_id in CLASS_MAPPING:
                boxes_abs.append((cls_id, x1, y1, x2, y2))
                orig_counts[cls_id] += 1

        tables = [b for b in boxes_abs if b[0] == 0]
        others = [b for b in boxes_abs if b[0] != 0]

        visited = set()
        clusters = []
        for i in range(len(tables)):
            if i in visited:
                continue
            cluster = [tables[i]]
            visited.add(i)
            queue = [tables[i]]
            while queue:
                curr = queue.pop(0)
                _, cx1, cy1, cx2, cy2 = curr
                for j in range(len(tables)):
                    if j in visited:
                        continue
                    _, tx1, ty1, tx2, ty2 = tables[j]
                    if (max(0, max(cx1,tx1) - min(cx2,tx2)) <= 15 and
                            max(0, max(cy1,ty1) - min(cy2,ty2)) <= 15):
                        cluster.append(tables[j])
                        visited.add(j)
                        queue.append(tables[j])
            clusters.append(cluster)

        for cluster in clusters:
            c_x1 = min(b[1] for b in cluster)
            c_y1 = min(b[2] for b in cluster)
            c_x2 = max(b[3] for b in cluster)
            c_y2 = max(b[4] for b in cluster)
            crop_img = img[c_y1:c_y2, c_x1:c_x2]
            internal_boxes = [(b[0], b[1]-c_x1, b[2]-c_y1, b[3]-c_x1, b[4]-c_y1)
                              for b in cluster]
            if len(cluster) > 1:
                connected_tables.append((crop_img, internal_boxes))
                connected_tables_full.append((img, (c_x1, c_y1, c_x2, c_y2), internal_boxes))
            else:
                crops_by_class[0].append((crop_img, internal_boxes))

        for b in others:
            cls_id, x1, y1, x2, y2 = b
            crops_by_class[cls_id].append(
                (img[y1:y2, x1:x2], [(cls_id, 0, 0, x2-x1, y2-y1)])
            )

    print(f"Noise crops       : {len(noise_crops)}")
    print(f"Connected tables  : {len(connected_tables_full)}")
    return crops_by_class, connected_tables, connected_tables_full, noise_crops, orig_counts


# --------------------------------------------------
# PAIRED SNAP
# --------------------------------------------------
def _pick_paired_snap(connected_tables_full, inner_rect, W, H, margin=6):
    ix2 = inner_rect["x2"]
    iy  = inner_rect["y"]
    iy2 = inner_rect["y2"]
    top_margin_h = iy - margin
    inner_h = iy2 - iy
    valid = []
    for entry in connected_tables_full:
        src_img, cluster_rect, internal_boxes = entry
        if len(internal_boxes) != 2:
            continue
        cx1, cy1 = cluster_rect[0], cluster_rect[1]
        abs_boxes = [
            (c, cx1 + bx1, cy1 + by1, cx1 + bx2, cy1 + by2)
            for c, bx1, by1, bx2, by2 in internal_boxes
        ]
        abs_boxes_sorted = sorted(abs_boxes, key=lambda b: b[2])
        outside_b = abs_boxes_sorted[0]
        inside_b  = abs_boxes_sorted[1]
        ow = outside_b[3] - outside_b[1]; oh = outside_b[4] - outside_b[2]
        iw = inside_b[3]  - inside_b[1];  ih = inside_b[4]  - inside_b[2]
        if oh <= top_margin_h and ow <= ix2 and ih <= inner_h and iw <= ix2:
            valid.append((entry, outside_b, inside_b))
    return valid

def place_paired_snap_tables(canvas, occupied, boxes_yolo, sheet_counts,
                              connected_tables_full, inner_rect, W, H, jitter=5):
    if sheet_counts.get(0, 0) >= TARGET_INSTANCES[0]:
        return False
    ix2 = inner_rect["x2"]
    iy  = inner_rect["y"]
    candidates = _pick_paired_snap(connected_tables_full, inner_rect, W, H)
    if not candidates:
        return False
    random.shuffle(candidates)
    for (entry, outside_b, inside_b) in candidates:
        src_img, cluster_rect, _ = entry
        _, ox1, oy1, ox2, oy2     = outside_b
        _, iix1, iiy1, iix2, iiy2 = inside_b
        out_crop = src_img[oy1:oy2, ox1:ox2].copy()
        in_crop  = src_img[iiy1:iiy2, iix1:iix2].copy()
        oh, ow = out_crop.shape[:2]
        ih, iw = in_crop.shape[:2]
        jx = random.randint(-jitter, jitter)
        jy = random.randint(-jitter, jitter)
        in_x  = int(np.clip(ix2 - iw + jx, 0, W - iw))
        in_y  = int(np.clip(iy + jy,        0, H - ih))
        out_x = int(np.clip(in_x + iw - ow, 0, W - ow))
        out_y = int(np.clip(in_y - oh,       0, H - oh))
        if out_y < 0 or out_x < 0:
            continue
        if in_y + ih > H or in_x + iw > W:
            continue
        if check_collision((in_x,  in_y,  in_x+iw,  in_y+ih),  0, occupied, OVERLAP_CLASSES): continue
        if check_collision((out_x, out_y, out_x+ow, out_y+oh), 0, occupied, OVERLAP_CLASSES): continue
        canvas[in_y:in_y+ih, in_x:in_x+iw]    = in_crop
        canvas[out_y:out_y+oh, out_x:out_x+ow] = out_crop
        occupied.append((in_x,  in_y,  in_x+iw,  in_y+ih,  0))
        occupied.append((out_x, out_y, out_x+ow, out_y+oh, 0))
        for (crop_x, crop_y, cw, ch) in [(in_x, in_y, iw, ih), (out_x, out_y, ow, oh)]:
            boxes_yolo.append((0,
                               (crop_x + cw / 2) / W,
                               (crop_y + ch / 2) / H,
                               cw / W, ch / H))
        sheet_counts[0] = sheet_counts.get(0, 0) + 1
        if sheet_counts[0] >= TARGET_INSTANCES[0]:
            break
    return True


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    random.seed(42)
    np.random.seed(42)

    if not INPUT_ROOT.exists():
        print(f"Error: Input folder {INPUT_ROOT} not found.")
        return

    setup_new_dataset()
    crops, connected_tables, connected_tables_full, noise_crops, orig_counts = load_crops_from_original()
    print(f"Original instance counts : {orig_counts}")
    print(f"Connected table groups   : {len(connected_tables)}")

    for cls_id in CLASS_MAPPING:
        if not crops[cls_id]:
            print(f"Warning: Class {cls_id} ({CLASS_MAPPING[cls_id]}) has 0 crops.")

    tight_counts = {k: 0 for k in CLASS_MAPPING}
    sheet_counts = {k: 0 for k in CLASS_MAPPING}
    mode_counts  = {"overlap": 0, "border": 0, "normal": 0}
    idx_tight = 0

    # Phase 1: Tight + loose crops — connected tables
    print("\nPhase 1: Tight + loose crops — connected tables...")
    for crop_img, internal_boxes in tqdm(connected_tables, desc="Connected tight"):
        h, w = crop_img.shape[:2]
        yolo_boxes = [(c, (bx1+bx2)/(2*w), (by1+by2)/(2*h),
                       (bx2-bx1)/w, (by2-by1)/h)
                      for c, bx1, by1, bx2, by2 in internal_boxes]
        for _rep in range(2):
            save_new_sample(crop_img, yolo_boxes, "aug_connected", idx_tight)
            idx_tight += 1
        for c, *_ in internal_boxes:
            tight_counts[c] += 1

    for src_img, cluster_rect, internal_boxes in tqdm(connected_tables_full, desc="Connected loose"):
        src_h, src_w = src_img.shape[:2]
        cx1, cy1, cx2, cy2 = cluster_rect
        cluster_w, cluster_h = cx2 - cx1, cy2 - cy1
        pad_left   = int(random.uniform(0.3, 0.8) * cluster_w)
        pad_right  = int(random.uniform(0.3, 0.8) * cluster_w)
        pad_top    = int(random.uniform(0.3, 0.8) * cluster_h)
        pad_bottom = int(random.uniform(0.3, 0.8) * cluster_h)
        lx1 = max(0, cx1 - pad_left);    ly1 = max(0, cy1 - pad_top)
        lx2 = min(src_w, cx2 + pad_right); ly2 = min(src_h, cy2 + pad_bottom)
        loose_crop = src_img[ly1:ly2, lx1:lx2]
        lh, lw = loose_crop.shape[:2]
        if lh == 0 or lw == 0:
            continue
        loose_boxes = []
        for (c, bx1, by1, bx2, by2) in internal_boxes:
            abs_x1 = cx1 + bx1 - lx1; abs_y1 = cy1 + by1 - ly1
            abs_x2 = cx1 + bx2 - lx1; abs_y2 = cy1 + by2 - ly1
            loose_boxes.append((c, (abs_x1+abs_x2)/(2*lw), (abs_y1+abs_y2)/(2*lh),
                                 (abs_x2-abs_x1)/lw, (abs_y2-abs_y1)/lh))
        for _rep in range(2):
            save_new_sample(loose_crop, loose_boxes, "aug_connected_loose", idx_tight)
            idx_tight += 1
        for c, *_ in internal_boxes:
            tight_counts[c] += 1

    # Phase 1b: Tight crops — single objects
    print("\nPhase 1b: Tight crops — single objects...")
    for cls_id, crop_list in crops.items():
        for crop_img, internal_boxes in crop_list:
            h, w = crop_img.shape[:2]
            yolo_boxes = [(c, (bx1+bx2)/(2*w), (by1+by2)/(2*h),
                           (bx2-bx1)/w, (by2-by1)/h)
                          for c, bx1, by1, bx2, by2 in internal_boxes]
            for _rep in range(2):
                save_new_sample(crop_img, yolo_boxes, "aug_tight", idx_tight)
                idx_tight += 1
            for c, *_ in internal_boxes:
                tight_counts[c] += 1

    # Phase 1c: Fake-stitched connected tables
    print("\nPhase 1c: Fake-stitched tight crops...")
    real_single_crops = list(crops[0])
    FAKE_TIGHT_COUNT = 500
    idx_fake = 0
    no_progress = 0
    while idx_fake < FAKE_TIGHT_COUNT:
        fake = create_fake_connected_table(real_single_crops)
        if fake is None:
            no_progress += 1
            if no_progress > 20:
                break
            continue
        no_progress = 0
        crop_img, internal_boxes = fake
        h, w = crop_img.shape[:2]
        if h == 0 or w == 0:
            continue
        yolo_boxes = [(c, (bx1+bx2)/(2*w), (by1+by2)/(2*h),
                       (bx2-bx1)/w, (by2-by1)/h)
                      for c, bx1, by1, bx2, by2 in internal_boxes]
        for _rep in range(2):
            save_new_sample(crop_img, yolo_boxes, "aug_fake_connected", idx_fake)
            idx_fake += 1
            idx_tight += 1
        for c, *_ in internal_boxes:
            tight_counts[c] += 1
    print(f"Fake-stitched tight crops saved: {idx_fake}")

    # Phase 2: Synthetic empty tables
    print(f"\nPhase 2: Synthetic empty tables (target: {TARGET_EMPTY_TABLES})...")
    idx_empty = 0
    empty_table_crops = []
    for _ in tqdm(range(TARGET_EMPTY_TABLES), desc="Empty tables"):
        crop_img, internal_boxes = generate_synthetic_empty_table()
        empty_table_crops.append((crop_img, internal_boxes))
        h, w = crop_img.shape[:2]
        yolo_boxes = [(c, (bx1+bx2)/(2*w), (by1+by2)/(2*h),
                       (bx2-bx1)/w, (by2-by1)/h)
                      for c, bx1, by1, bx2, by2 in internal_boxes]
        save_new_sample(crop_img, yolo_boxes, "aug_empty", idx_empty)
        idx_empty += 1
        idx_tight += 1
        for c, *_ in internal_boxes:
            tight_counts[c] += 1

    crops[0].extend(empty_table_crops)
    _empty_ids = {id(c) for c in empty_table_crops}
    real_table_crops_only = [c for c in crops[0] if id(c) not in _empty_ids]
    print(f"Empty table files saved: {idx_empty}")
    print(f"Total tight crop files : {idx_tight}")
    print(f"Tight counts (not toward sheet target): {tight_counts}")

    # Pre-load drawing frame assets once
    frame_meta, frame_images = load_frame_assets()
    frames_available = (
        frame_meta is not None and
        any(frame_images.get(o) for o in ("horizontal", "vertical"))
    )
    if not frames_available:
        print("No drawing frame assets found — all sheets will use plain white canvas.")

    # Phase 3: Composite sheets
    print(f"\nPhase 3: Composite sheets (targets: { {CLASS_MAPPING[c]: TARGET_INSTANCES[c] for c in CLASS_MAPPING} })...")
    idx_sheet = 0
    idx_frame_sheets = 0
    no_placement_streak = 0
    MAX_EMPTY_STREAK = 2000  # raised to handle larger targets without early bail-out

    connected_pool = list(connected_tables)
    random.shuffle(connected_pool)

    def pick_table_crop():
        if connected_pool and random.random() < 0.6:
            return random.choice(connected_pool)
        real = crops[0]
        if real:
            if random.random() < 0.5 and len(real) >= 2:
                fake = create_fake_connected_table(real)
                if fake:
                    return fake
            return random.choice(real)
        return generate_synthetic_empty_table()

    # Loop until ALL classes have hit their targets (no secondary table brake)
    while any(sheet_counts[c] < TARGET_INSTANCES[c] for c in CLASS_MAPPING):
        if no_placement_streak >= MAX_EMPTY_STREAK:
            print("Warning: Hit max empty-sheet streak, exiting early.")
            break

        needy = [c for c in CLASS_MAPPING if sheet_counts[c] < TARGET_INSTANCES[c]]
        if not needy:
            break

        # Weight by deficit: most-behind class gets chosen more often
        deficits = [TARGET_INSTANCES[c] - sheet_counts[c] for c in needy]
        target_cls = random.choices(needy, weights=deficits, k=1)[0]

        use_frame  = frames_available and (random.random() < FRAME_CANVAS_PROB)
        inner_rect = None

        if use_frame:
            result = pick_random_frame(frame_meta, frame_images)
            if result is None:
                use_frame = False

        if use_frame:
            canvas, W, H, inner_rect = result
            idx_frame_sheets += 1
        else:
            W, H = get_random_canvas_size()
            canvas = np.full((H, W, 3), 255, dtype=np.uint8)

        boxes_yolo = []
        occupied   = []

        if noise_crops and random.random() < NOISE_PER_SHEET_PROB:
            paste_noise_elements(canvas, occupied, noise_crops, W, H)

        if target_cls in OVERLAP_CLASSES:
            sheet_mode = _weighted_choice(OVERLAP_MODE_WEIGHTS)
        else:
            sheet_mode = "normal"

        is_cluster_mode = (target_cls in OVERLAP_CLASSES) and (
            sheet_mode == "overlap" or random.random() > 0.5)
        cluster_center  = (_ri(0, W), _ri(0, H)) if is_cluster_mode else None
        primary_processed = False

        num_objects = random.randint(3, MAX_OBJECTS_PER_SHEET)
        for i in range(num_objects):
            current_needy = [c for c in CLASS_MAPPING if sheet_counts[c] < TARGET_INSTANCES[c]]
            if not current_needy:
                break

            if not primary_processed:
                cls = target_cls
                primary_processed = True
            else:
                if is_cluster_mode:
                    valid = [c for c in OVERLAP_CLASSES if c in current_needy]
                    cls = random.choice(valid) if valid else random.choice(current_needy)
                else:
                    cls = (target_cls if random.random() > 0.5 and target_cls in current_needy
                           else random.choice(current_needy))

            if cls == 0:
                obj_img, internal_boxes = pick_table_crop()
            else:
                if not crops[cls]:
                    continue
                obj_img, internal_boxes = random.choice(crops[cls])

            proposed = {}
            for (c, *_) in internal_boxes:
                proposed[c] = proposed.get(c, 0) + 1
            if any(sheet_counts[c] + qty > TARGET_INSTANCES[c] for c, qty in proposed.items()):
                continue

            h_obj, w_obj = obj_img.shape[:2]
            if h_obj >= H or w_obj >= W:
                continue

            if use_frame and inner_rect is not None:
                ix, iy   = inner_rect["x"],  inner_rect["y"]
                ix2, iy2 = inner_rect["x2"], inner_rect["y2"]
                if w_obj > (ix2 - ix) or h_obj > (iy2 - iy):
                    continue
                x_min_p, y_min_p, x_max_p, y_max_p = ix, iy, ix2, iy2
            else:
                x_min_p, y_min_p, x_max_p, y_max_p = 0, 0, W, H

            for _ in range(50):
                if cls in OVERLAP_CLASSES and sheet_mode != "normal":
                    if sheet_mode == "border":
                        x, y = place_border(W, H, w_obj, h_obj,
                                            x_min_p, y_min_p, x_max_p, y_max_p)
                    else:
                        x, y = place_overlapping(W, H, w_obj, h_obj, occupied,
                                                 x_min_p, y_min_p, x_max_p, y_max_p)
                elif cls != 0:
                    large_on_canvas = any(
                        (ex2-ex1)*(ey2-ey1) > w_obj*h_obj * 1.5
                        for ex1, ey1, ex2, ey2, ec in occupied if ec in OVERLAP_CLASSES
                    )
                    if large_on_canvas and (w_obj * h_obj) < 40000:
                        x, y = place_small_under_large(W, H, w_obj, h_obj, occupied,
                                                       x_min_p, y_min_p, x_max_p, y_max_p)
                    elif is_cluster_mode and cls in OVERLAP_CLASSES:
                        x = int(np.clip(
                            cluster_center[0] + int(random.gauss(0, w_obj * 1.5)),
                            x_min_p, max(x_min_p, x_max_p - w_obj)))
                        y = int(np.clip(
                            cluster_center[1] + int(random.gauss(0, h_obj * 1.5)),
                            y_min_p, max(y_min_p, y_max_p - h_obj)))
                    else:
                        x = _ri(x_min_p, max(x_min_p, x_max_p - w_obj))
                        y = _ri(y_min_p, max(y_min_p, y_max_p - h_obj))
                else:
                    x = _ri(x_min_p, max(x_min_p, x_max_p - w_obj))
                    y = _ri(y_min_p, max(y_min_p, y_max_p - h_obj))

                iou_limit = 0.18 if sheet_mode == "overlap" and cls in OVERLAP_CLASSES else 0.20
                if check_collision((x, y, x+w_obj, y+h_obj), cls, occupied,
                                   OVERLAP_CLASSES, max_iou=iou_limit):
                    continue

                canvas[y:y+h_obj, x:x+w_obj] = obj_img
                occupied.append((x, y, x+w_obj, y+h_obj, cls))
                for (c, bx1, by1, bx2, by2) in internal_boxes:
                    ax1, ay1 = x + bx1, y + by1
                    ax2, ay2 = x + bx2, y + by2
                    boxes_yolo.append((c,
                                       (ax1+ax2)/(2*W), (ay1+ay2)/(2*H),
                                       (ax2-ax1)/W, (ay2-ay1)/H))
                    sheet_counts[c] += 1
                if cls in OVERLAP_CLASSES:
                    mode_counts[sheet_mode] += 1
                break

        # Outside-frame tables on frame sheets
        if use_frame and inner_rect is not None:
            if random.random() < OUTSIDE_FRAME_PROB:
                if random.random() < PAIRED_SNAP_PROB and connected_tables_full:
                    place_paired_snap_tables(
                        canvas, occupied, boxes_yolo, sheet_counts,
                        connected_tables_full, inner_rect, W, H
                    )
                elif crops[0]:
                    n_bonus = random.randint(1, 2)
                    for _ in range(n_bonus):
                        if sheet_counts[0] >= TARGET_INSTANCES[0]:
                            break
                        if connected_pool and random.random() < 0.6:
                            bonus_img, bonus_iboxes = random.choice(connected_pool)
                        elif real_table_crops_only:
                            bonus_img, bonus_iboxes = random.choice(real_table_crops_only)
                        else:
                            continue
                        bh, bw = bonus_img.shape[:2]
                        if bh >= H or bw >= W:
                            continue
                        result_margin = place_table_in_margin(W, H, bonus_img, bonus_iboxes, inner_rect)
                        if result_margin is None:
                            continue
                        fit_img, fit_boxes, bx, by = result_margin
                        fbh, fbw = fit_img.shape[:2]
                        if check_collision((bx, by, bx+fbw, by+fbh), 0, occupied, OVERLAP_CLASSES):
                            continue
                        canvas[by:by+fbh, bx:bx+fbw] = fit_img
                        occupied.append((bx, by, bx+fbw, by+fbh, 0))
                        for (c, bbx1, bby1, bbx2, bby2) in fit_boxes:
                            ax1, ay1 = bx + bbx1, by + bby1
                            ax2, ay2 = bx + bbx2, by + bby2
                            boxes_yolo.append((c,
                                               (ax1+ax2)/(2*W), (ay1+ay2)/(2*H),
                                               (ax2-ax1)/W, (ay2-ay1)/H))
                            sheet_counts[c] += 1

        if boxes_yolo:
            save_new_sample(canvas, boxes_yolo, "aug_sheet", idx_sheet)
            idx_sheet += 1
            no_placement_streak = 0
        else:
            no_placement_streak += 1

    # Final report
    print(f"\n{'='*60}")
    print(f" Dataset created at: {OUTPUT_ROOT}")
    print(f"{'='*60}")
    print(f"{'Class':<22} {'ID':<6} {'Tight':<10} {'Sheets':<10} {'Target':<10} Status")
    print(f"{'-'*60}")
    for cls_id, cls_name in CLASS_MAPPING.items():
        sc = sheet_counts[cls_id]
        tc = tight_counts[cls_id]
        status = "OK" if sc >= TARGET_INSTANCES[cls_id] else f"UNDER by {TARGET_INSTANCES[cls_id] - sc}"
        print(f"{cls_name:<22} {cls_id:<6} {tc:<10} {sc:<10} {TARGET_INSTANCES[cls_id]:<10} {status}")
    print(f"{'-'*60}")
    print(f"Sheet files saved      : {idx_sheet}")
    print(f" of which frame sheets : {idx_frame_sheets}")
    print(f"Tight crop files saved : {idx_tight}")
    total_oc = sum(sheet_counts[c] for c in OVERLAP_CLASSES)
    print(f"\nPlacement mode breakdown (overlap-class instances on sheets):")
    for mode, cnt in mode_counts.items():
        pct = 100 * cnt / max(1, total_oc)
        print(f"  {mode:<10}: {cnt:>7} ({pct:.1f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()