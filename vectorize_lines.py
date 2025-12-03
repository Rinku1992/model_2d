# vectorize_lines.py
"""
Vectorization: skeleton -> Hough -> dynamic gap-based merging.

This version uses:
    - Very low minLineLength (tunable from CLI)
    - Data-driven merge thresholds using percentile gaps per orientation
"""

from __future__ import annotations
import os
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import math

Line = Tuple[int, int, int, int]

# -------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------
def _angle_deg(line: Line) -> float:
    x1, y1, x2, y2 = line
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def classify_orientation(line: Line, angle_tol_deg: float = 5.0) -> str:
    """
    classify line as Horizontal (H), Vertical (V), or Other (O)
    """
    ang = abs(_angle_deg(line))
    ang = ang % 180

    if ang <= angle_tol_deg or abs(ang - 180) <= angle_tol_deg:
        return "H"
    if abs(ang - 90) <= angle_tol_deg:
        return "V"
    return "O"


def detect_lines(
    skel: np.ndarray,
    min_line_length: int,
    max_line_gap: int = 10,
    debug_dir: Optional[str] = None,
) -> List[Line]:
    """
    Run HoughLinesP on the 1px skeleton.
    """
    lines_p = cv2.HoughLinesP(
        skel,
        rho=1,
        theta=np.pi/180,
        threshold=25,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    lines: List[Line] = []
    if lines_p is not None:
        for [[x1, y1, x2, y2]] in lines_p:
            lines.append((int(x1), int(y1), int(x2), int(y2)))

    if debug_dir:
        h, w = skel.shape
        vis = np.full((h, w, 3), 255, np.uint8)
        for (x1, y1, x2, y2) in lines:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.imwrite(os.path.join(debug_dir, "05_lines_raw.png"), vis)

    return lines


# -------------------------------------------------------------
# Gap collection & percentiles
# -------------------------------------------------------------
def _project(line: Line, orient: str) -> Tuple[int, int]:
    x1, y1, x2, y2 = line
    if orient == "H":
        return (min(x1, x2), max(x1, x2))
    else:
        return (min(y1, y2), max(y1, y2))


def _axis(line: Line, orient: str) -> float:
    x1, y1, x2, y2 = line
    return (y1 + y2) * 0.5 if orient == "H" else (x1 + x2) * 0.5


def compute_gap_distribution(lines: List[Line], orient: str) -> List[int]:
    """
    Compute positive gaps along projection axis between consecutive segments
    on the same row/column.
    """
    if not lines:
        return []

    # sort by axis coord first, then by start of projection
    lines_sorted = sorted(lines, key=lambda L: (_axis(L, orient), _project(L, orient)[0]))

    gaps = []
    current_axis = None
    bucket: List[Line] = []

    def flush_bucket():
        if len(bucket) <= 1:
            return
        # sort by start of projection
        bucket_sorted = sorted(bucket, key=lambda L: _project(L, orient)[0])
        prev = bucket_sorted[0]
        for nxt in bucket_sorted[1:]:
            a1, b1 = _project(prev, orient)
            a2, b2 = _project(nxt, orient)
            gap = a2 - b1
            if gap > 0:
                gaps.append(gap)
            prev = nxt

    for ln in lines_sorted:
        ax = _axis(ln, orient)
        if current_axis is None or abs(ax - current_axis) <= 3:
            bucket.append(ln)
            current_axis = ax if current_axis is None else current_axis
        else:
            flush_bucket()
            bucket = [ln]
            current_axis = ax
    flush_bucket()

    return gaps


def compute_gap_threshold(
    gaps: List[int],
    percentile: float,
    fallback: int = 3,
    min_cap: int = 1,
    max_cap: int = 50,
) -> int:
    """
    Compute percentile gap threshold with caps.
    """
    if len(gaps) == 0:
        return fallback

    g = np.array(gaps)
    thr = int(np.percentile(g, percentile))

    thr = max(min_cap, min(thr, max_cap))
    return thr


# -------------------------------------------------------------
# Merging logic
# -------------------------------------------------------------
def merge_lines_dynamic(
    lines: List[Line],
    hv_gap_percentile: float,
    diag_gap_percentile: float,
    angle_tol_deg: float = 5.0,
    debug_dir: Optional[str] = None,
    shape: Optional[Tuple[int, int]] = None,
    fixed_gap_override: Optional[int] = None,
) -> List[Line]:

    # 1. Classify
    H = []
    V = []
    O = []
    for ln in lines:
        o = classify_orientation(ln, angle_tol_deg)
        if o == "H": H.append(ln)
        elif o == "V": V.append(ln)
        else: O.append(ln)

    # 2. Compute gap distributions
    gaps_H = compute_gap_distribution(H, "H")
    gaps_V = compute_gap_distribution(V, "V")
    gaps_O = compute_gap_distribution(O, "H")  # diagonal: arbitrary axis

    # 3. Compute thresholds
    if fixed_gap_override is not None:
        gap_H = gap_V = gap_O = fixed_gap_override
    else:
        gap_H = compute_gap_threshold(gaps_H, hv_gap_percentile)
        gap_V = compute_gap_threshold(gaps_V, hv_gap_percentile)
        gap_O = compute_gap_threshold(gaps_O, diag_gap_percentile)

    # 4. Merge
    def merge_group(group: List[Line], orient: str, gapthr: int) -> List[Line]:
        if not group:
            return []
        # sort by axis -> start
        sorted_g = sorted(group, key=lambda L: (_axis(L, orient), _project(L, orient)[0]))

        merged: List[Line] = []
        bucket: List[Line] = []
        current_axis = None

        def flush_bucket():
            if not bucket:
                return
            bucket_sorted = sorted(bucket, key=lambda L: _project(L, orient)[0])
            cur_start, cur_end = _project(bucket_sorted[0], orient)

            for ln in bucket_sorted[1:]:
                a, b = _project(ln, orient)
                if a <= cur_end + gapthr:
                    cur_end = max(cur_end, b)
                else:
                    if orient == "H":
                        y = int(round(_axis(bucket_sorted[0], orient)))
                        merged.append((cur_start, y, cur_end, y))
                    else:
                        x = int(round(_axis(bucket_sorted[0], orient)))
                        merged.append((x, cur_start, x, cur_end))
                    cur_start, cur_end = a, b

            if orient == "H":
                y = int(round(_axis(bucket_sorted[0], orient)))
                merged.append((cur_start, y, cur_end, y))
            else:
                x = int(round(_axis(bucket_sorted[0], orient)))
                merged.append((x, cur_start, x, cur_end))

        for ln in sorted_g:
            ax = _axis(ln, orient)
            if current_axis is None or abs(ax - current_axis) <= 3:
                bucket.append(ln)
                current_axis = ax if current_axis is None else current_axis
            else:
                flush_bucket()
                bucket = [ln]
                current_axis = ax
        flush_bucket()

        return merged

    merged_H = merge_group(H, "H", gap_H)
    merged_V = merge_group(V, "V", gap_V)

    # For diagonal lines — either do strict merging or no merging
    # Here we treat them like "tiny segments" and merge only with tiny gapthr
    merged_O = merge_group(O, "H", gap_O)

    merged_all = merged_H + merged_V + merged_O

    if debug_dir and shape is not None:
        h, w = shape
        vis = np.full((h, w, 3), 255, np.uint8)
        for (x1, y1, x2, y2) in merged_all:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.imwrite(os.path.join(debug_dir, "06_lines_merged.png"), vis)

    return merged_all


# -------------------------------------------------------------
# Render
# -------------------------------------------------------------
def render_lines(lines: List[Line], shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    canvas = np.full((h, w), 255, np.uint8)
    for (x1, y1, x2, y2) in lines:
        cv2.line(canvas, (x1, y1), (x2, y2), 0, 1)
    return canvas
