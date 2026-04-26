#!/usr/bin/env python3
"""
Train a binary classifier for undirected relations between bounding boxes and
export raw/postprocessed error overlays.

Output structure:

images/errors_raw/
    FP/
    FN/
    MIXED/

images/errors_postprocessed/
    FP/
    FN/
    MIXED/
    differences/

Notes:
- One output image per source image, saved directly inside the respective folder.
- No summary.txt files.
- No nested per-image folders.
- "differences" contains overlays only for rows changed by postprocessing:
    - green line = added relation (0 -> 1)
    - red line   = removed relation (1 -> 0)
"""

import json
import re
import math
import itertools
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


SCHEMATIC_LABEL = "schem"
ICON_LABEL = "icon"
TEXT_LABELS = {"text"}
VISUAL_LABELS = {SCHEMATIC_LABEL, ICON_LABEL}

ARROW_LABELS = {
    "abl_arrow", "abr_arrow", "ang_arrow", "atl_arrow", "atr_arrow",
    "b_arrow", "circular_arrow", "dhb_arrow", "dht_arrow", "dlb_arrow",
    "dlt_arrow", "drb_arrow", "drt_arrow", "dv_arrow", "dvl_arrow",
    "dvr_arrow", "l_arrow", "r_arrow", "sv_arrow", "t_arrow",
}

ALLOWED_LABELS = ({SCHEMATIC_LABEL, ICON_LABEL} | TEXT_LABELS | ARROW_LABELS)
ALLOWED_LABELS_SORTED = sorted(ALLOWED_LABELS)

NEAR_DIST_THRESHOLD = 0.2
NEAR_IOU_THRESHOLD = 0.0
FAR_NEG_KEEP_PROB = 0.10

URL_PREFIX = "/data/local-files/?d="
CSV_PATH = Path("pairs_dataset.csv")
ANNOTATION_JSON = Path("result_v12.json")
BASE_IMG_DIR = Path("images")

RAW_ERROR_ROOT = BASE_IMG_DIR / "errors_raw"
POST_ERROR_ROOT = BASE_IMG_DIR / "errors_postprocessed"

RAW_FP_DIR = RAW_ERROR_ROOT / "FP"
RAW_FN_DIR = RAW_ERROR_ROOT / "FN"
RAW_MIXED_DIR = RAW_ERROR_ROOT / "MIXED"

POST_FP_DIR = POST_ERROR_ROOT / "FP"
POST_FN_DIR = POST_ERROR_ROOT / "FN"
POST_MIXED_DIR = POST_ERROR_ROOT / "MIXED"
POST_DIFF_DIR = POST_ERROR_ROOT / "differences"


with open(ANNOTATION_JSON, "r", encoding="utf-8") as f:
    anno_data = json.load(f)

print(f"Loaded {len(anno_data)} annotated images.")


def extract_filename(raw_img: str) -> str:
    if raw_img.startswith(URL_PREFIX):
        raw_img = raw_img[len(URL_PREFIX):]
    return Path(raw_img).name


def safe_stem(raw_img: str) -> str:
    return Path(extract_filename(raw_img)).stem


def build_image_id_map(samples):
    mapping = {}
    for sample in samples:
        raw_img = sample.get("data", {}).get("image", "")
        if not raw_img:
            continue
        fname = extract_filename(raw_img)
        mapping[fname] = sample.get("id", "")
    return mapping


IMAGE_ID_MAP = build_image_id_map(anno_data)


def get_image_identifier(raw_img: str) -> str:
    fname = extract_filename(raw_img)
    stem = Path(fname).stem
    internal_id = IMAGE_ID_MAP.get(fname) or IMAGE_ID_MAP.get(stem) or ""
    return str(internal_id) if internal_id else stem


def make_output_name(raw_img: str, suffix: str) -> str:
    identifier = get_image_identifier(raw_img)
    stem = safe_stem(raw_img)
    if identifier != stem:
        return f"{identifier}_{stem}_{suffix}.png"
    return f"{stem}_{suffix}.png"


def compute_features(bA, bB, iw, ih):
    def to_abs(b):
        x = b["x"] / 100 * iw
        y = b["y"] / 100 * ih
        w = b["width"] / 100 * iw
        h = b["height"] / 100 * ih
        return x + w / 2, y + h / 2, w, h, x, y

    cxA, cyA, wA, hA, x1A, y1A = to_abs(bA)
    cxB, cyB, wB, hB, x1B, y1B = to_abs(bB)

    dist = math.sqrt((cxA - cxB) ** 2 + (cyA - cyB) ** 2)
    diag = math.sqrt(iw ** 2 + ih ** 2)

    ix1 = max(x1A, x1B)
    iy1 = max(y1A, y1B)
    ix2 = min(x1A + wA, x1B + wB)
    iy2 = min(y1A + hA, y1B + hB)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = wA * hA + wB * hB - inter
    iou = inter / union if union > 0 else 0.0

    areaA = wA * hA
    areaB = wB * hB

    return {
        "d_norm": dist / diag,
        "x_A_norm": cxA / iw,
        "y_A_norm": cyA / ih,
        "x_B_norm": cxB / iw,
        "y_B_norm": cyB / ih,
        "cx_min_norm": min(cxA, cxB) / iw,
        "cx_max_norm": max(cxA, cxB) / iw,
        "cy_min_norm": min(cyA, cyB) / ih,
        "cy_max_norm": max(cyA, cyB) / ih,
        "w_min_norm": min(wA, wB) / iw,
        "w_max_norm": max(wA, wB) / iw,
        "h_min_norm": min(hA, hB) / ih,
        "h_max_norm": max(hA, hB) / ih,
        "iou": iou,
        "delta_x_norm": abs(cxA - cxB) / iw,
        "delta_y_norm": abs(cyA - cyB) / ih,
        "area_A_norm": areaA / (iw * ih),
        "area_B_norm": areaB / (iw * ih),
        "area_min_norm": min(areaA, areaB) / (iw * ih),
        "area_max_norm": max(areaA, areaB) / (iw * ih),
        "area_ratio": max(areaA, areaB) / (min(areaA, areaB) + 1e-6),
        "aspect_A": wA / (hA + 1e-6),
        "aspect_B": wB / (hB + 1e-6),
        "aspect_min": min(wA / (hA + 1e-6), wB / (hB + 1e-6)),
        "aspect_max": max(wA / (hA + 1e-6), wB / (hB + 1e-6)),
        "w_ratio": max(wA, wB) / (min(wA, wB) + 1e-6),
        "h_ratio": max(hA, hB) / (min(hA, hB) + 1e-6),
    }


def build_rect_index(samples):
    image_to_results = {}
    for sample in samples:
        ann_src = sample.get("predictions") or sample.get("annotations") or []
        if not ann_src:
            continue
        results = ann_src[0].get("result", [])
        raw_img = sample.get("data", {}).get("image", "")
        if not raw_img:
            continue
        fname = extract_filename(raw_img)
        rects = {r["id"]: r for r in results if r["type"] == "rectanglelabels"}
        if rects:
            image_to_results[fname] = rects
    return image_to_results


def _abs_box(rect):
    b = rect["value"]
    ow = rect["original_width"]
    oh = rect["original_height"]
    x = b["x"] / 100 * ow
    y = b["y"] / 100 * oh
    w = b["width"] / 100 * ow
    h = b["height"] / 100 * oh
    return (x, y, x + w, y + h)


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _fully_contains(outer, inner):
    return (
        inner[0] >= outer[0]
        and inner[1] >= outer[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def _pt_inside(box, pt):
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def _partner_id(row, this_id):
    return row["id_B"] if row["id_A"] == this_id else row["id_A"]


image_items = []
for sample in anno_data:
    ann_src = sample.get("predictions") or sample.get("annotations") or []
    if not ann_src:
        continue
    results = ann_src[0].get("result", [])
    if not any(r["type"] == "rectanglelabels" for r in results):
        continue
    image_items.append(sample)

train_imgs, test_imgs = train_test_split(image_items, test_size=0.2, random_state=42)


def build_pairs(samples, rng_seed=42, sample_negatives=True):
    rng = np.random.default_rng(rng_seed)
    rows = []

    for sample in samples:
        ann_src = sample.get("predictions") or sample.get("annotations") or []
        if not ann_src:
            continue
        results = ann_src[0].get("result", [])

        boxes = {r["id"]: r for r in results if r["type"] == "rectanglelabels"}
        if not boxes:
            continue

        def box_label(r):
            return r["value"]["rectanglelabels"][0]

        allowed_ids = [bid for bid, r in boxes.items() if box_label(r) in ALLOWED_LABELS]
        if not allowed_ids:
            continue

        pos_pairs = set()
        for r in results:
            if r["type"] != "relation":
                continue
            f, t = r["from_id"], r["to_id"]
            if f in allowed_ids and t in allowed_ids:
                pos_pairs.add(tuple(sorted([f, t])))

        widths = {boxes[bid]["original_width"] for bid in allowed_ids}
        heights = {boxes[bid]["original_height"] for bid in allowed_ids}
        if len(widths) > 1 or len(heights) > 1:
            print(f"[WARN] Inconsistent dimensions in {sample['data'].get('image', '')} — skipping.")
            continue

        iw = widths.pop()
        ih = heights.pop()
        img = sample["data"].get("image", "")

        for idA, idB in itertools.combinations(sorted(allowed_ids), 2):
            bA = boxes[idA]["value"]
            bB = boxes[idB]["value"]
            lA = bA["rectanglelabels"][0]
            lB = bB["rectanglelabels"][0]

            if lA in TEXT_LABELS and lB in TEXT_LABELS:
                continue

            feats = compute_features(bA, bB, iw, ih)
            is_pos = tuple(sorted([idA, idB])) in pos_pairs

            if not is_pos and sample_negatives:
                near = (feats["d_norm"] <= NEAR_DIST_THRESHOLD) or (feats["iou"] > NEAR_IOU_THRESHOLD)
                if (not near) and (rng.random() > FAR_NEG_KEEP_PROB):
                    continue

            label_1, label_2 = sorted([lA, lB])

            rows.append({
                "image": img,
                "id_A": idA,
                "id_B": idB,
                "label_1": label_1,
                "label_2": label_2,
                "orig_label_A": lA,
                "orig_label_B": lB,
                **feats,
                "target": int(is_pos),
            })

    return pd.DataFrame(rows)


if CSV_PATH.exists():
    print(f"[INFO] {CSV_PATH} already exists — loading from disk (delete to rebuild).")
    df = pd.read_csv(CSV_PATH)
else:
    print("Building training pairs...")
    df_train = build_pairs(train_imgs, rng_seed=42, sample_negatives=True)
    print("Building test pairs...")
    df_test = build_pairs(test_imgs, rng_seed=123, sample_negatives=False)

    df_train = df_train.assign(split="train")
    df_test = df_test.assign(split="test")
    df = pd.concat([df_train, df_test], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved -> {CSV_PATH}")


NUMERIC_FEATURES = [
    "d_norm",
    "cx_min_norm", "cx_max_norm",
    "cy_min_norm", "cy_max_norm",
    "w_min_norm", "w_max_norm",
    "h_min_norm", "h_max_norm",
    "iou",
    "delta_x_norm", "delta_y_norm",
    "area_A_norm", "area_B_norm",
    "area_min_norm", "area_max_norm",
    "area_ratio",
    "aspect_A", "aspect_B",
    "aspect_min", "aspect_max",
    "w_ratio", "h_ratio",
]

CATEGORICAL_FEATURES = ["label_1", "label_2"]

ohe = OneHotEncoder(
    categories=[ALLOWED_LABELS_SORTED, ALLOWED_LABELS_SORTED],
    handle_unknown="ignore",
    sparse_output=False,
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", ohe, CATEGORICAL_FEATURES),
    ],
    remainder="drop",
)

ALL_FEAT = NUMERIC_FEATURES + CATEGORICAL_FEATURES
train_mask = df["split"] == "train"
test_mask = df["split"] == "test"

X_train_raw = df.loc[train_mask, ALL_FEAT]
X_test_raw = df.loc[test_mask, ALL_FEAT]
y_train = df.loc[train_mask, "target"].values
y_test = df.loc[test_mask, "target"].values

X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
    X_train_raw,
    y_train,
    test_size=0.1,
    random_state=42,
    stratify=y_train,
)

train_neg = (y_tr == 0).sum()
train_pos = (y_tr == 1).sum()
scale_pos_weight = train_neg / max(train_pos, 1)
print(f"\nscale_pos_weight set to {scale_pos_weight:.2f}")

X_tr_scaled = preprocessor.fit_transform(X_tr_raw)
X_val_scaled = preprocessor.transform(X_val_raw)
X_test_scaled = preprocessor.transform(X_test_raw)

clf = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr",
    early_stopping_rounds=20,
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)

clf.fit(
    X_tr_scaled,
    y_tr,
    eval_set=[(X_val_scaled, y_val)],
    verbose=50,
)

print(f"\nBest XGBoost iteration: {clf.best_iteration}")

ohe_feat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES).tolist()
FEATURE_NAMES = NUMERIC_FEATURES + ohe_feat_names

y_val_prob = clf.predict_proba(X_val_scaled)[:, 1]
y_test_prob = clf.predict_proba(X_test_scaled)[:, 1]

precisions_v, recalls_v, thresholds_v = precision_recall_curve(y_val, y_val_prob)
f1_val = 2 * (precisions_v * recalls_v) / (precisions_v + recalls_v + 1e-8)
best_idx = np.argmax(f1_val[:-1]) if len(thresholds_v) > 0 else 0
best_threshold = float(thresholds_v[best_idx]) if len(thresholds_v) > 0 else 0.5

y_pred_optimal = (y_test_prob >= best_threshold).astype(int)

print("\n── Classification Report (raw predictions) ──")
print(classification_report(
    y_test,
    y_pred_optimal,
    labels=[0, 1],
    target_names=["No Relation", "Related"],
    digits=3,
    zero_division=0,
))
print(f"Accuracy : {accuracy_score(y_test, y_pred_optimal):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_optimal, zero_division=0):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_test_prob):.4f}")
print(f"PR-AUC   : {average_precision_score(y_test, y_test_prob):.4f}")

importances = pd.Series(clf.feature_importances_, index=FEATURE_NAMES)
importances.sort_values(ascending=False).rename("xgb_importance").to_csv("xgb_feature_importance.csv")

perm = permutation_importance(
    clf,
    X_test_scaled,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="average_precision",
    n_jobs=-1,
)

perm_df = pd.DataFrame({
    "feature": FEATURE_NAMES,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False)
perm_df.to_csv("permutation_importance.csv", index=False)

preprocessor_final = clone(preprocessor)
preprocessor_final.fit(X_tr_raw)

df_test_pairs = df.loc[test_mask].copy()
df_test_pairs["y_true"] = y_test
df_test_pairs["y_prob"] = y_test_prob
df_test_pairs["y_pred"] = y_pred_optimal
df_test_pairs["relationship_status"] = df_test_pairs["y_true"].map({1: "associated", 0: "unassociated"})


def compute_error_type(y_true_series, y_pred_series):
    out = pd.Series("TN", index=y_true_series.index, dtype=object)
    out.loc[(y_true_series == 1) & (y_pred_series == 0)] = "FN"
    out.loc[(y_true_series == 0) & (y_pred_series == 1)] = "FP"
    out.loc[(y_true_series == 1) & (y_pred_series == 1)] = "TP"
    return out


df_test_pairs["error_type"] = compute_error_type(df_test_pairs["y_true"], df_test_pairs["y_pred"])
df_test_pairs.to_csv("pairs_with_predictions.csv", index=False)
print("Saved -> pairs_with_predictions.csv")

image_to_results = build_rect_index(anno_data)


def apply_hard_rules(df_pairs: pd.DataFrame, image_to_results: dict) -> pd.DataFrame:
    df_pairs = df_pairs.copy()
    df_pairs["y_pred_post"] = df_pairs["y_pred"].astype(int)
    df_pairs["rule_applied"] = ""

    stats = {"text_forced": 0, "text_pruned": 0, "arrow_forced": 0}

    for raw_img, img_idx in df_pairs.groupby("image").groups.items():
        fname = extract_filename(raw_img)
        rects = image_to_results.get(fname)
        if rects is None:
            continue

        local_ids = set(df_pairs.loc[img_idx, "id_A"]) | set(df_pairs.loc[img_idx, "id_B"])
        box_cache = {}
        label_cache = {}

        for obj_id in local_ids:
            if obj_id in rects:
                box_cache[obj_id] = _abs_box(rects[obj_id])
                label_cache[obj_id] = rects[obj_id]["value"]["rectanglelabels"][0]

        visual_ids = [
            i for i in local_ids
            if label_cache.get(i) in VISUAL_LABELS and i in box_cache
        ]

        inside_visual = {}
        for obj_id in local_ids:
            lbl = label_cache.get(obj_id)
            if lbl not in (TEXT_LABELS | ARROW_LABELS):
                continue
            if obj_id not in box_cache:
                continue

            obj_box = box_cache[obj_id]
            obj_ctr = _center(obj_box)

            for vid in visual_ids:
                if vid == obj_id:
                    continue
                vbox = box_cache[vid]

                if lbl in TEXT_LABELS and _fully_contains(vbox, obj_box):
                    inside_visual[obj_id] = vid
                    break

                if lbl in ARROW_LABELS and _pt_inside(vbox, obj_ctr):
                    inside_visual[obj_id] = vid
                    break

        if not inside_visual:
            continue

        for obj_id in inside_visual:
            lbl = label_cache.get(obj_id)

            obj_mask = (df_pairs["image"] == raw_img) & (
                (df_pairs["id_A"] == obj_id) | (df_pairs["id_B"] == obj_id)
            )

            obj_rows = df_pairs.loc[obj_mask]
            if obj_rows.empty:
                continue

            pred_on_idx = obj_rows.index[obj_rows["y_pred_post"] == 1]

            def _partner_ok(row):
                pid = _partner_id(row, obj_id)
                return label_cache.get(pid) in (VISUAL_LABELS | ARROW_LABELS)

            cands = obj_rows[obj_rows.apply(_partner_ok, axis=1)]
            if cands.empty:
                cands = obj_rows

            if len(pred_on_idx) == 0:
                best_i = cands["y_prob"].idxmax()
                df_pairs.at[best_i, "y_pred_post"] = 1
                if lbl in TEXT_LABELS:
                    df_pairs.at[best_i, "rule_applied"] = "force_text_inside_one"
                    stats["text_forced"] += 1
                else:
                    df_pairs.at[best_i, "rule_applied"] = "force_arrow_inside_one"
                    stats["arrow_forced"] += 1

            elif lbl in TEXT_LABELS and len(pred_on_idx) >= 2:
                best_i = df_pairs.loc[pred_on_idx, "y_prob"].idxmax()
                drop_ix = [i for i in pred_on_idx if i != best_i]
                df_pairs.loc[drop_ix, "y_pred_post"] = 0
                df_pairs.loc[drop_ix, "rule_applied"] = "prune_text_inside_to_one"
                stats["text_pruned"] += len(drop_ix)

    df_pairs["error_type_post"] = compute_error_type(df_pairs["y_true"], df_pairs["y_pred_post"])

    print("\n── Hard-rule post-processing stats ──")
    for k, v in stats.items():
        print(f" {k}: {v}")
    print(f" total rows changed: {(df_pairs['rule_applied'] != '').sum()}")

    return df_pairs


df_test_pairs = apply_hard_rules(df_test_pairs, image_to_results)
df_test_pairs.to_csv("pairs_with_predictions_postprocessed.csv", index=False)
print("Saved -> pairs_with_predictions_postprocessed.csv")


def ensure_dirs():
    for path in [
        RAW_FP_DIR, RAW_FN_DIR, RAW_MIXED_DIR,
        POST_FP_DIR, POST_FN_DIR, POST_MIXED_DIR, POST_DIFF_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def get_pair_boxes(rects, row):
    idA, idB = row["id_A"], row["id_B"]
    if idA not in rects or idB not in rects:
        return None

    rA = rects[idA]["value"]
    rB = rects[idB]["value"]
    orig_w = rects[idA]["original_width"]
    orig_h = rects[idA]["original_height"]

    def to_abs_box(b):
        bx = b["x"] / 100 * orig_w
        by = b["y"] / 100 * orig_h
        bw = b["width"] / 100 * orig_w
        bh = b["height"] / 100 * orig_h
        return (bx, by, bx + bw, by + bh)

    return to_abs_box(rA), to_abs_box(rB)


def draw_pair(draw, row, rects, line_color, label_text):
    boxes = get_pair_boxes(rects, row)
    if boxes is None:
        return False

    boxA, boxB = boxes

    draw.rectangle(boxA, outline="blue", width=3)
    draw.rectangle(boxB, outline="orange", width=3)

    cxA = (boxA[0] + boxA[2]) / 2
    cyA = (boxA[1] + boxA[3]) / 2
    cxB = (boxB[0] + boxB[2]) / 2
    cyB = (boxB[1] + boxB[3]) / 2

    draw.line((cxA, cyA, cxB, cyB), fill=line_color, width=3)
    draw.text(
        (min(boxA[0], boxB[0]), max(0, min(boxA[1], boxB[1]) - 15)),
        label_text,
        fill="white",
    )
    return True


def save_grouped_error_overlays(df_pairs, error_col, root_fp, root_fn, root_mixed, name_suffix):
    saved = {"FP": 0, "FN": 0, "MIXED": 0}

    err_df = df_pairs[df_pairs[error_col].isin(["FP", "FN"])].copy()
    if err_df.empty:
        return saved

    for raw_img, image_rows in err_df.groupby("image"):
        fname = extract_filename(raw_img)
        img_path = BASE_IMG_DIR / fname
        if not img_path.exists():
            print(f"[DEBUG] Missing image: {img_path}")
            continue

        rects = image_to_results.get(fname)
        if rects is None:
            print(f"[DEBUG] No rect index for: {fname}")
            continue

        try:
            im = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[DEBUG] PIL error {img_path}: {e}")
            continue

        draw = ImageDraw.Draw(im)

        present_types = set(image_rows[error_col].unique())
        if present_types == {"FP"}:
            out_dir = root_fp
            bucket = "FP"
        elif present_types == {"FN"}:
            out_dir = root_fn
            bucket = "FN"
        else:
            out_dir = root_mixed
            bucket = "MIXED"

        n_drawn = 0
        sort_ascending = True if bucket == "FN" else False
        for _, row in image_rows.sort_values("y_prob", ascending=sort_ascending).iterrows():
            err = row[error_col]
            if err not in {"FP", "FN"}:
                continue
            line_color = "red" if err == "FP" else "green"
            label_text = (
                f"{err} "
                f"target={int(row['y_true'])} "
                f"pred={int(row['y_pred_post']) if error_col == 'error_type_post' else int(row['y_pred'])} "
                f"p={row['y_prob']:.2f} "
                f"[{row['orig_label_A']}--{row['orig_label_B']}]"
            )
            ok = draw_pair(draw, row, rects, line_color, label_text)
            if ok:
                n_drawn += 1

        if n_drawn == 0:
            continue

        out_path = out_dir / make_output_name(raw_img, name_suffix)
        im.save(out_path)
        saved[bucket] += 1
        print(f"[DEBUG] Saved {bucket} overlay: {out_path}")

    return saved


def save_difference_overlays(df_pairs, out_dir):
    saved = 0
    changed_df = df_pairs[df_pairs["y_pred"] != df_pairs["y_pred_post"]].copy()
    if changed_df.empty:
        return saved

    for raw_img, image_rows in changed_df.groupby("image"):
        fname = extract_filename(raw_img)
        img_path = BASE_IMG_DIR / fname
        if not img_path.exists():
            print(f"[DEBUG] Missing image: {img_path}")
            continue

        rects = image_to_results.get(fname)
        if rects is None:
            print(f"[DEBUG] No rect index for: {fname}")
            continue

        try:
            im = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[DEBUG] PIL error {img_path}: {e}")
            continue

        draw = ImageDraw.Draw(im)
        n_drawn = 0

        for _, row in image_rows.sort_values("y_prob", ascending=False).iterrows():
            if row["y_pred"] == row["y_pred_post"]:
                continue

            added = row["y_pred"] == 0 and row["y_pred_post"] == 1
            removed = row["y_pred"] == 1 and row["y_pred_post"] == 0
            if not (added or removed):
                continue

            line_color = "green" if added else "red"
            action = "ADDED" if added else "REMOVED"
            label_text = (
                f"{action} "
                f"true={int(row['y_true'])} "
                f"raw={int(row['y_pred'])} "
                f"post={int(row['y_pred_post'])} "
                f"p={row['y_prob']:.2f} "
                f"[{row['orig_label_A']}--{row['orig_label_B']}]"
            )

            ok = draw_pair(draw, row, rects, line_color, label_text)
            if ok:
                n_drawn += 1

        if n_drawn == 0:
            continue

        out_path = out_dir / make_output_name(raw_img, "differences")
        im.save(out_path)
        saved += 1
        print(f"[DEBUG] Saved differences overlay: {out_path}")

    return saved


ensure_dirs()

print("\nRendering raw error overlays...")
raw_saved = save_grouped_error_overlays(
    df_pairs=df_test_pairs,
    error_col="error_type",
    root_fp=RAW_FP_DIR,
    root_fn=RAW_FN_DIR,
    root_mixed=RAW_MIXED_DIR,
    name_suffix="raw_errors",
)

print("\nRendering postprocessed error overlays...")
post_saved = save_grouped_error_overlays(
    df_pairs=df_test_pairs,
    error_col="error_type_post",
    root_fp=POST_FP_DIR,
    root_fn=POST_FN_DIR,
    root_mixed=POST_MIXED_DIR,
    name_suffix="postprocessed_errors",
)

print("\nRendering postprocessed differences overlays...")
diff_saved = save_difference_overlays(
    df_pairs=df_test_pairs,
    out_dir=POST_DIFF_DIR,
)

y_true = df_test_pairs["y_true"].values
y_pred_raw = df_test_pairs["y_pred"].values
y_pred_post = df_test_pairs["y_pred_post"].values
y_prob = df_test_pairs["y_prob"].values

print("\n── Classification Report (RAW predictions) ──")
print(classification_report(
    y_true,
    y_pred_raw,
    labels=[0, 1],
    target_names=["No Relation", "Related"],
    digits=3,
    zero_division=0,
))
print(f"Accuracy : {accuracy_score(y_true, y_pred_raw):.4f}")
print(f"Precision: {precision_score(y_true, y_pred_raw, zero_division=0):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_true, y_prob):.4f}")
print(f"PR-AUC   : {average_precision_score(y_true, y_prob):.4f}")

print("\n── Classification Report (POSTPROCESSED predictions) ──")
print(classification_report(
    y_true,
    y_pred_post,
    labels=[0, 1],
    target_names=["No Relation", "Related"],
    digits=3,
    zero_division=0,
))
print(f"Accuracy : {accuracy_score(y_true, y_pred_post):.4f}")
print(f"Precision: {precision_score(y_true, y_pred_post, zero_division=0):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_true, y_prob):.4f}")
print(f"PR-AUC   : {average_precision_score(y_true, y_prob):.4f}")

print("\n── Error image counts ──")
print(
    "RAW model pictures  -> "
    f"FP: {raw_saved['FP']} | FN: {raw_saved['FN']} | MIXED: {raw_saved['MIXED']} | "
    f"TOTAL: {sum(raw_saved.values())}"
)
print(
    "POST model pictures -> "
    f"FP: {post_saved['FP']} | FN: {post_saved['FN']} | MIXED: {post_saved['MIXED']} | "
    f"DIFFERENCES: {diff_saved} | TOTAL_ERRORS: {sum(post_saved.values())}"
)

joblib.dump(
    {
        "preprocessor": preprocessor_final,
        "clf": clf,
        "threshold": best_threshold,
        "feature_names": FEATURE_NAMES,
        "numeric_feats": NUMERIC_FEATURES,
        "categorical_feats": CATEGORICAL_FEATURES,
        "allowed_labels": ALLOWED_LABELS_SORTED,
        "undirected": True,
    },
    "pair_classifier.joblib",
)
print("Saved -> pair_classifier.joblib")