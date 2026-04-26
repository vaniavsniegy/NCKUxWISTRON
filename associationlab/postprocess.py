import json
from collections import defaultdict

# rearrange.py — top of file
ARROW_LABELS = {
    "abl_arrow", "abr_arrow", "ang_arrow", "atl_arrow", "atr_arrow",
    "b_arrow", "circular_arrow", "dhb_arrow", "dht_arrow", "dlb_arrow",
    "dlt_arrow", "drb_arrow", "drt_arrow", "dv_arrow", "dvl_arrow",
    "dvr_arrow", "l_arrow", "r_arrow", "sv_arrow", "t_arrow",
}
TEXT_LABELS = {"text"}
TABLE_LABELS = set()  # no table labels in new schema

SCHEMATIC_LBL = "schem"
ICON_LBL = "icon"
OTHER_LBL = "other"
CONTAINER_LBLS = {SCHEMATIC_LBL, ICON_LBL, OTHER_LBL}

def get_label(item):
    try:
        return item["value"]["rectanglelabels"][0]
    except Exception:
        return None

def bbox_percent(item):
    v = item["value"]
    return (v["x"], v["y"], v["x"] + v["width"], v["y"] + v["height"])

def centroid(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def dist(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5

def is_inside(outer, inner, threshold=0.7):
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner

    x_left = max(ox1, ix1)
    y_top = max(oy1, iy1)
    x_right = min(ox2, ix2)
    y_bottom = min(oy2, iy2)

    if x_right < x_left or y_bottom < y_top:
        return False

    inter_area = (x_right - x_left) * (y_bottom - y_top)
    inner_area = (ix2 - ix1) * (iy2 - iy1)

    if inner_area == 0:
        return False

    return (inter_area / inner_area) >= threshold

def smallest_container(item_bbox, candidates_dict):
    """Finds the smallest area container that overlaps the item by >= threshold."""
    matches = [
        (i, bbox_percent(c))
        for i, c in candidates_dict.items()
        if is_inside(bbox_percent(c), item_bbox)
    ]
    if not matches:
        return None
    return min(
        matches,
        key=lambda m: (m[1][2] - m[1][0]) * (m[1][3] - m[1][1])
    )[0]

def process_task(task):
    results = task["annotations"][0]["result"]
    items = {r["id"]: r for r in results if r["type"] == "rectanglelabels"}

    schematics = {i: items[i] for i in items if get_label(items[i]) == SCHEMATIC_LBL}
    icons = {i: items[i] for i in items if get_label(items[i]) == ICON_LBL}
    others = {i: items[i] for i in items if get_label(items[i]) == OTHER_LBL}
    all_containers = {**schematics, **icons, **others}

    def is_table_item(i):
        return get_label(items[i]) in TABLE_LABELS

    # --- IDENTIFY OUTSIDE ARROWS (no sufficient overlap with any container) ---
    outside_arrows = set()
    for i_id, item in items.items():
        if get_label(item) in ARROW_LABELS:
            if not smallest_container(bbox_percent(item), all_containers):
                outside_arrows.add(i_id)

    adj = defaultdict(set)
    preserved_relations = []
    manual_overrides = defaultdict(set)

    # --- PHASE 1: RELATIONSHIP CLASSIFICATION ---
    for r in results:
        if r["type"] != "relation":
            continue
        a, b = r["from_id"], r["to_id"]

        # Do not consider relations involving outside arrows
        if a in outside_arrows or b in outside_arrows:
            continue

        # Do not consider relations whose endpoints are table cells
        if a in items and is_table_item(a):
            continue
        if b in items and is_table_item(b):
            continue

        lbl_a, lbl_b = get_label(items.get(a)), get_label(items.get(b))

        if lbl_a in CONTAINER_LBLS or lbl_b in CONTAINER_LBLS:
            # Keep manual container relations
            preserved_relations.append(r)
            if lbl_b in CONTAINER_LBLS:
                manual_overrides[a].add(b)
            if lbl_a in CONTAINER_LBLS:
                manual_overrides[b].add(a)
        else:
            # FIX: include text↔text links in adjacency so chained texts
            # reach their arrow during BFS clustering.
            # Post-Filter 3 will strip text↔text edges from the final output.
            adj[a].add(b)
            adj[b].add(a)

    # Only arrows + texts are eligible, no outside arrows, no table cells
    eligible = {
        i for i in items
        if get_label(items[i]) in (ARROW_LABELS | TEXT_LABELS)
        and i not in outside_arrows
        and not is_table_item(i)
    }

    # --- PHASE 2: CLUSTERING ---
    visited, components = set(), []
    for node in eligible:
        if node not in visited:
            comp, queue = set(), [node]
            while queue:
                n = queue.pop()
                if n in visited or n not in eligible:
                    continue
                visited.add(n)
                comp.add(n)
                for nb in adj[n]:
                    if nb in eligible:
                        queue.append(nb)
            if comp:
                components.append(comp)

    new_relations = []

    def add_rel(f, t):
        new_relations.append({
            "from_id": f,
            "to_id": t,
            "type": "relation",
            "direction": "right",
            "id": f"gen_{len(new_relations)}"
        })

    def find_parent(id, allow_icon=True):
        # Manual container links always win
        if manual_overrides[id]:
            return next(iter(manual_overrides[id]))
        bbox = bbox_percent(items[id])

        # Prefer schematics (large symbols)
        p = smallest_container(bbox, schematics)
        if p:
            return p

        # Then icons, if allowed
        if allow_icon:
            p_icon = smallest_container(bbox, icons)
            if p_icon:
                return p_icon

        # Finally, fall back to "other" if present
        return smallest_container(bbox, others)

    def get_schematic_parent(item_id):
        return smallest_container(bbox_percent(items[item_id]), schematics)

    def find_closest_arrow_to_parent(comp_arrows):
        """Pick the arrow whose centroid is closest to its parent container's centroid."""
        best, best_dist = comp_arrows[0], float("inf")
        for a_id in comp_arrows:
            parent_id = find_parent(a_id, allow_icon=False)
            if parent_id is None:
                continue
            d = dist(
                centroid(bbox_percent(items[a_id])),
                centroid(bbox_percent(items[parent_id]))
            )
            if d < best_dist:
                best, best_dist = a_id, d
        return best

    # --- PHASE 3: REWIRING ---
    associated = set()
    for comp in components:
        comp_arrows = list(comp & {i for i in items if get_label(items[i]) in ARROW_LABELS})
        comp_texts = list(comp & {i for i in items if get_label(items[i]) in TEXT_LABELS})

        if comp_arrows:
            root_arrow = find_closest_arrow_to_parent(comp_arrows)

            # Each text points to root arrow
            for t in comp_texts:
                add_rel(t, root_arrow)

            # Non-root arrows point to root arrow
            for a in comp_arrows:
                if a != root_arrow:
                    add_rel(a, root_arrow)

            # Root arrow points to its parent container
            if not manual_overrides[root_arrow]:
                parent = find_parent(root_arrow, allow_icon=False)
                if parent:
                    add_rel(root_arrow, parent)
        else:
            # No arrow in cluster — each text independently finds its parent
            for t in comp_texts:
                if not manual_overrides[t]:
                    parent = find_parent(t, allow_icon=True)
                    if parent:
                        add_rel(t, parent)

        associated.update(comp)

    # --- PHASE 4: ORPHAN ASSIGNMENT (Texts Only) ---
    for item_id in eligible - associated:
        if get_label(items[item_id]) in ARROW_LABELS:
            continue
        if not manual_overrides[item_id]:
            parent = find_parent(item_id, allow_icon=True)
            if parent:
                add_rel(item_id, parent)

    # --- POST-FILTER 1: Remove cross-schematic relations ---
    def same_schematic(f, t):
        # If either endpoint is a container, allow it
        if get_label(items.get(f)) in CONTAINER_LBLS or get_label(items.get(t)) in CONTAINER_LBLS:
            return True
        sf, st = get_schematic_parent(f), get_schematic_parent(t)
        if sf is None or st is None:
            return True
        return sf == st

    # --- POST-FILTER 2: Remove all table-related relations ---
    def involves_table(r):
        return (
            get_label(items.get(r["from_id"])) in TABLE_LABELS or
            get_label(items.get(r["to_id"])) in TABLE_LABELS
        )

    new_relations = [
        r for r in new_relations
        if same_schematic(r["from_id"], r["to_id"]) and not involves_table(r)
    ]
    preserved_relations = [
        r for r in preserved_relations
        if not involves_table(r)
    ]

    # --- POST-FILTER 3: Remove all text-text relations ---
    def is_text_text_relation(r):
        lbl_from = get_label(items.get(r["from_id"]))
        lbl_to = get_label(items.get(r["to_id"]))
        return lbl_from in TEXT_LABELS and lbl_to in TEXT_LABELS

    new_relations = [r for r in new_relations if not is_text_text_relation(r)]
    preserved_relations = [r for r in preserved_relations if not is_text_text_relation(r)]

    # --- CLEAN & ASSEMBLE ---
    clean_task = {"data": task.get("data", {}), "annotations": []}

    non_rel_metadata = []
    for r in results:
        if r["type"] not in ["rectanglelabels", "relation"]:
            r.pop("origin", None)
            non_rel_metadata.append(r)

    clean_items = []
    for item in items.values():
        item_copy = item.copy()
        item_copy.pop("origin", None)
        clean_items.append(item_copy)

    clean_ann = {
        "result": clean_items + non_rel_metadata + preserved_relations + new_relations,
        "was_cancelled": False,
        "ground_truth": True
    }
    clean_task["annotations"].append(clean_ann)
    return clean_task

# --- EXECUTION ---
with open("project-13-at-2026-04-08-07-22-9fbae150.json", "r") as f:
    data = json.load(f)

final_base = [process_task(task) for task in data]

with open("result_v12.json", "w") as f:
    json.dump(final_base, f)

print(f"Done. {len(final_base)} tasks processed. Saved as: result_v12.json")
