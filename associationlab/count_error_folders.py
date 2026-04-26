#!/usr/bin/env python3
from pathlib import Path
import argparse


def count_direct_subfolders(path: Path) -> int:
    return sum(1 for p in path.iterdir() if p.is_dir()) if path.exists() else 0


def count_files_recursive(path: Path) -> int:
    return sum(1 for p in path.rglob('*') if p.is_file()) if path.exists() else 0


def get_direct_folder_names(path: Path) -> set[str]:
    return {p.name for p in path.iterdir() if p.is_dir()} if path.exists() else set()


def report_root(root: Path) -> None:
    fp_dir    = root / "FP"
    fn_dir    = root / "FN"
    mixed_dir = root / "MIXED"

    print(f"\nroot: {root.resolve()}")

    for branch in [fp_dir, fn_dir, mixed_dir]:
        if branch.exists() and branch.is_dir():
            print(f"  {branch.name}_image_folders:            {count_direct_subfolders(branch)}")
            print(f"  {branch.name}_total_files_recursive:    {count_files_recursive(branch)}")
        else:
            print(f"  {branch.name}_image_folders:            0")
            print(f"  {branch.name}_total_files_recursive:    0")

    fp_names    = get_direct_folder_names(fp_dir)
    fn_names    = get_direct_folder_names(fn_dir)
    mixed_names = get_direct_folder_names(mixed_dir)

    both          = fp_names & fn_names
    fp_only       = fp_names - fn_names - mixed_names
    fn_only       = fn_names - fp_names - mixed_names
    unique_images = fp_names | fn_names | mixed_names

    print(f"  unique_image_folders_across_FP_FN_MIXED: {len(unique_images)}")
    print(f"  image_folders_in_both_FP_and_FN:         {len(both)}")
    print(f"  image_folders_only_in_FP:                {len(fp_only)}")
    print(f"  image_folders_only_in_FN:                {len(fn_only)}")
    print(f"  image_folders_only_in_MIXED:             {len(mixed_names - fp_names - fn_names)}")

    print(f"\n  Per-branch counts:")
    branches = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name) if root.exists() else []
    for child in branches:
        n_folders = count_direct_subfolders(child)
        n_files   = count_files_recursive(child)
        print(f"    {child.name}: {n_folders} image folders, {n_files} total files")

    total_fp_errors = len(fp_names)
    total_fn_errors = len(fn_names)
    total_mixed     = len(mixed_names)
    total_all       = len(unique_images)
    print(f"\n  TOTAL error images (FP+FN+MIXED, deduplicated): {total_all}")
    print(f"    of which FP:    {total_fp_errors}")
    print(f"    of which FN:    {total_fn_errors}")
    print(f"    of which MIXED: {total_mixed}")


def report_postprocess(root: Path) -> None:
    """
    Report for images/postprocess/ which has a different structure:
      postprocess/
        <rule_name>/          e.g. force_text_inside_one/
          <image_folder>/
            summary.txt
            <id>_postprocess.png
        mixed_rules/
          <image_folder>/
    """
    print(f"\nroot: {root.resolve()}  [postprocess layout]")

    if not root.exists():
        print("  (directory does not exist)")
        return

    rule_dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)

    if not rule_dirs:
        print("  (no rule sub-folders found)")
        return

    all_image_folders: set[str] = set()
    rule_summary: list[tuple[str, int, int]] = []  # (rule, n_image_folders, n_files)

    for rule_dir in rule_dirs:
        img_folders = get_direct_folder_names(rule_dir)
        n_files     = count_files_recursive(rule_dir)
        rule_summary.append((rule_dir.name, len(img_folders), n_files))
        all_image_folders |= img_folders

    # Per-rule breakdown
    print(f"  Per-rule counts:")
    for rule_name, n_folders, n_files in rule_summary:
        print(f"    {rule_name}:")
        print(f"      image_folders:   {n_folders}")
        print(f"      total_files:     {n_files}")

    # Overlap between rules
    rule_folder_sets = {rd.name: get_direct_folder_names(rd) for rd in rule_dirs}
    print(f"\n  Cross-rule overlap (images touched by more than one rule):")
    rule_names = [rd.name for rd in rule_dirs]
    for i in range(len(rule_names)):
        for j in range(i + 1, len(rule_names)):
            overlap = rule_folder_sets[rule_names[i]] & rule_folder_sets[rule_names[j]]
            if overlap:
                print(f"    {rule_names[i]} ∩ {rule_names[j]}: {len(overlap)}")

    print(f"\n  TOTAL unique image folders across all rules: {len(all_image_folders)}")
    print(f"  TOTAL files across all rules:                {count_files_recursive(root)}")

    # Pngs vs summaries
    n_png  = sum(1 for p in root.rglob("*.png"))
    n_txt  = sum(1 for p in root.rglob("summary.txt"))
    print(f"  overlay PNGs:    {n_png}")
    print(f"  summary.txt:     {n_txt}")


def main():
    parser = argparse.ArgumentParser(
        description="Count error/postprocess folders and files under output roots.",
    )
    parser.add_argument(
        "roots",
        nargs="*",
        default=["images/errors", "images/error_inside", "images/postprocess"],
        help=(
            "One or more root directories to report. "
            "Default: images/errors  images/error_inside  images/postprocess"
        ),
    )
    args = parser.parse_args()

    roots = [Path(r) for r in args.roots]

    error_roots      = []
    postprocess_roots = []

    for root in roots:
        if not root.exists():
            print(f"\n[SKIP] {root.resolve()} does not exist.")
            continue
        # Detect layout: postprocess roots have rule-name sub-dirs, not FP/FN/MIXED
        children = {p.name for p in root.iterdir() if p.is_dir()}
        if children & {"FP", "FN", "MIXED"}:
            error_roots.append(root)
        else:
            postprocess_roots.append(root)

    for root in error_roots:
        report_root(root)

    for root in postprocess_roots:
        report_postprocess(root)

    # Cross-root comparison when exactly two error roots are provided
    if len(error_roots) == 2:
        r1, r2 = error_roots
        names1 = (
            get_direct_folder_names(r1 / "FP") |
            get_direct_folder_names(r1 / "FN") |
            get_direct_folder_names(r1 / "MIXED")
        )
        names2 = (
            get_direct_folder_names(r2 / "FP") |
            get_direct_folder_names(r2 / "FN") |
            get_direct_folder_names(r2 / "MIXED")
        )
        print(f"\n── Cross-root comparison ──")
        print(f"  {r1.name} total unique images: {len(names1)}")
        print(f"  {r2.name} total unique images: {len(names2)}")
        print(f"  in both roots:                 {len(names1 & names2)}")
        print(f"  only in {r1.name}:             {len(names1 - names2)}")
        print(f"  only in {r2.name}:             {len(names2 - names1)}")


if __name__ == "__main__":
    main()
