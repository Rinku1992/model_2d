# pipeline.py
"""
Deterministic floorplan cleaner with:
    - low minLineLength
    - dynamic percentile merge thresholds
"""

from __future__ import annotations
import os
import argparse
import cv2

from preprocessing import load_grayscale, preprocess_floorplan
from floorplan_cleaner.skeletonize_walls_old import skeletonize_and_prune
from floorplan_cleaner.vectorize_lines_old import (
    detect_lines,
    merge_lines_dynamic,
    render_lines,
)


def run_pipeline(
    input_path: str,
    output_path: str,
    debug_dir: str,
    min_line_length: int,
    hv_gap_percentile: float,
    diag_gap_percentile: float,
    fixed_gap_thresh: int | None,
):
    os.makedirs(debug_dir, exist_ok=True)

    # Step 1: Preprocess
    gray = load_grayscale(input_path)
    binary = preprocess_floorplan(gray, debug_dir)

    # Step 2: Skeleton
    skel_raw, skel_pruned = skeletonize_and_prune(
        binary,
        spur_iterations=1,
        debug_dir=debug_dir
    )

    # 🚨 FIX: convert skeleton to uint8 {0,255}
    skel_pruned_u8 = (skel_pruned > 0).astype("uint8") * 255

    cv2.imwrite(os.path.join(debug_dir, "04_skeleton_pruned_fixed.png"), skel_pruned_u8)

    # Step 3: Hough
    lines_raw = detect_lines(
        skel_pruned_u8,
        min_line_length=min_line_length,
        max_line_gap=10,
        debug_dir=debug_dir,
    )

    # Step 4: Merge dynamically
    merged = merge_lines_dynamic(
        lines_raw,
        hv_gap_percentile=hv_gap_percentile,
        diag_gap_percentile=diag_gap_percentile,
        angle_tol_deg=5.0,
        debug_dir=debug_dir,
        shape=skel_pruned_u8.shape,
        fixed_gap_override=fixed_gap_thresh,
    )

    # Step 5: Render result
    final = render_lines(merged, skel_pruned_u8.shape)
    cv2.imwrite(output_path, final)
    print(f"✅ Saved final cleaned plan → {output_path}")


def main():
    parser = argparse.ArgumentParser("Deterministic percentile-based floorplan cleaner")

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--debug-dir", required=True)

    parser.add_argument("--min-line-length", type=int, default=10)
    parser.add_argument("--hv-gap-percentile", type=float, default=5.0)
    parser.add_argument("--diag-gap-percentile", type=float, default=2.5)
    parser.add_argument("--fixed-gap-thresh", type=int, default=None)

    args = parser.parse_args()

    run_pipeline(
        args.input,
        args.output,
        args.debug_dir,
        args.min_line_length,
        args.hv_gap_percentile,
        args.diag_gap_percentile,
        args.fixed_gap_thresh,
    )


if __name__ == "__main__":
    main()
