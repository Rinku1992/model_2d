"""wall_stitcher.py

Post-process a pruned skeleton to produce continuous, 1-pixel-wide walls.

Usage:
    python wall_stitcher.py --input output/04_skeleton_pruned.png --output output/05_skeleton_stitched.png
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import cv2
import numpy as np

Coord = Tuple[int, int]  # (row, col)


@dataclass
class Segment:
    path: List[Coord]
    orientation: str | None  # "horizontal", "vertical", or None


NEIGHBORS_8: Tuple[Coord, ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def load_binary_mask(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read input image: {path}")
    # Normalize to {0,1}
    _, binary = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary.astype(np.uint8)


def flag_if_not_single_pixel(mask: np.ndarray) -> bool:
    """
    Return True if any 2x2 fully-foreground block exists (implies thickness > 1).
    """
    kernel = np.ones((2, 2), dtype=np.uint8)
    counts = cv2.filter2D(mask, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    thick = np.any(counts == 4)
    if thick:
        # find one example to report
        coords = np.argwhere(counts == 4)
        sample = tuple(int(v) for v in coords[0]) if coords.size else None
        sys.stderr.write(
            f"[HITL] Detected wall thickness >1px near pixel (row, col)={sample}. "
            "Please confirm input skeleton before stitching.\n"
        )
    return thick


def neighbors(coord: Coord, foreground: Set[Coord]) -> List[Coord]:
    r, c = coord
    result = []
    for dr, dc in NEIGHBORS_8:
        cand = (r + dr, c + dc)
        if cand in foreground:
            result.append(cand)
    return result


def build_degree_map(foreground: Set[Coord]) -> Dict[Coord, int]:
    return {pix: len(neighbors(pix, foreground)) for pix in foreground}


def trace_segments(foreground: Set[Coord]) -> List[List[Coord]]:
    degree = build_degree_map(foreground)
    nodes = {p for p, deg in degree.items() if deg != 2}

    visited_edges: Set[frozenset[Coord]] = set()
    segments: List[List[Coord]] = []

    def mark_path(path: Sequence[Coord]) -> None:
        for i in range(len(path) - 1):
            visited_edges.add(frozenset((path[i], path[i + 1])))

    # Walk from nodes first (endpoints / junctions)
    for node in nodes:
        for nb in neighbors(node, foreground):
            edge = frozenset((node, nb))
            if edge in visited_edges:
                continue
            path = [node]
            prev, current = node, nb
            while True:
                path.append(current)
                nbrs = neighbors(current, foreground)
                nbrs = [n for n in nbrs if n != prev]
                if degree.get(current, 0) != 2 or not nbrs:
                    break
                next_pixel = nbrs[0]
                prev, current = current, next_pixel
            mark_path(path)
            segments.append(path)

    # Handle closed loops of degree==2 (no nodes)
    for start in list(foreground):
        for nb in neighbors(start, foreground):
            edge = frozenset((start, nb))
            if edge in visited_edges:
                continue
            path = [start]
            prev, current = start, nb
            while True:
                path.append(current)
                nbrs = neighbors(current, foreground)
                nbrs = [n for n in nbrs if n != prev]
                if not nbrs:
                    break
                next_pixel = nbrs[0]
                edge_next = frozenset((current, next_pixel))
                if edge_next in visited_edges:
                    break
                prev, current = current, next_pixel
                if current == start:
                    path.append(current)
                    break
            mark_path(path)
            segments.append(path)

    return segments


def classify_segment(path: Sequence[Coord]) -> str | None:
    if len(path) < 2:
        return None
    ys = [p[0] for p in path]
    xs = [p[1] for p in path]
    dy = max(ys) - min(ys)
    dx = max(xs) - min(xs)
    if dx == dy == 0:
        return None
    if dx >= dy * 1.2:
        return "horizontal"
    if dy >= dx * 1.2:
        return "vertical"
    return None


def extract_segments(mask: np.ndarray) -> List[Segment]:
    foreground = {(int(r), int(c)) for r, c in zip(*np.nonzero(mask))}
    raw_paths = trace_segments(foreground)
    segments: List[Segment] = []
    for path in raw_paths:
        orient = classify_segment(path)
        segments.append(Segment(path=list(path), orientation=orient))
    return segments


def canonical_line_value(values: Iterable[int]) -> int:
    arr = np.fromiter(values, dtype=int)
    if arr.size == 0:
        return 0
    return int(np.median(arr))


def merge_axis_aligned(segments: List[Segment]) -> Tuple[List[Tuple[int, int, int, int]], List[Segment]]:
    """
    Merge horizontal/vertical segments separately.

    Returns
    -------
    merged_lines : list of (x1, y1, x2, y2) tuples
    untouched    : list of segments that were not classified
    """
    horizontals: List[Dict] = []  # type: ignore[var-annotated]
    verticals: List[Dict] = []  # type: ignore[var-annotated]
    unknown: List[Segment] = []

    for seg in segments:
        ys = [p[0] for p in seg.path]
        xs = [p[1] for p in seg.path]
        if seg.orientation == "horizontal":
            horizontals.append({
                "y": canonical_line_value(ys),
                "x_min": min(xs),
                "x_max": max(xs),
            })
        elif seg.orientation == "vertical":
            verticals.append({
                "x": canonical_line_value(xs),
                "y_min": min(ys),
                "y_max": max(ys),
            })
        else:
            unknown.append(seg)

    merged: List[Tuple[int, int, int, int]] = []

    # Horizontal merging by y band (±1 drift)
    horizontals.sort(key=lambda h: h["y"])
    bands: List[List[Dict[str, int]]] = []
    for h in horizontals:
        placed = False
        for band in bands:
            band_y = canonical_line_value([e["y"] for e in band])
            if abs(h["y"] - band_y) <= 1:
                band.append(h)
                placed = True
                break
        if not placed:
            bands.append([h])

    for band in bands:
        band_y = canonical_line_value([h["y"] for h in band])
        band.sort(key=lambda h: h["x_min"])
        current = band[0].copy()
        for nxt in band[1:]:
            if nxt["x_min"] <= current["x_max"] + 1:
                current["x_max"] = max(current["x_max"], nxt["x_max"])
                current["y"] = canonical_line_value([current["y"], nxt["y"]])
            else:
                merged.append((current["x_min"], band_y, current["x_max"], band_y))
                current = nxt.copy()
        merged.append((current["x_min"], band_y, current["x_max"], band_y))

    # Vertical merging by x band (±1 drift)
    verticals.sort(key=lambda v: v["x"])
    v_bands: List[List[Dict[str, int]]] = []
    for v in verticals:
        placed = False
        for band in v_bands:
            band_x = canonical_line_value([e["x"] for e in band])
            if abs(v["x"] - band_x) <= 1:
                band.append(v)
                placed = True
                break
        if not placed:
            v_bands.append([v])

    for band in v_bands:
        band_x = canonical_line_value([v["x"] for v in band])
        band.sort(key=lambda v: v["y_min"])
        current = band[0].copy()
        for nxt in band[1:]:
            if nxt["y_min"] <= current["y_max"] + 1:
                current["y_max"] = max(current["y_max"], nxt["y_max"])
                current["x"] = canonical_line_value([current["x"], nxt["x"]])
            else:
                merged.append((band_x, current["y_min"], band_x, current["y_max"]))
                current = nxt.copy()
        merged.append((band_x, current["y_min"], band_x, current["y_max"]))

    return merged, unknown


def draw_lines(shape: Tuple[int, int], lines: List[Tuple[int, int, int, int]]) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.uint8)
    for x1, y1, x2, y2 in lines:
        cv2.line(canvas, (x1, y1), (x2, y2), color=1, thickness=1)
    return canvas


def fill_orthogonal_gaps(mask: np.ndarray) -> np.ndarray:
    """Fill 1-pixel gaps at T/corner junctions."""
    up = np.pad(mask[:-1, :], ((1, 0), (0, 0)), mode="constant")
    down = np.pad(mask[1:, :], ((0, 1), (0, 0)), mode="constant")
    left = np.pad(mask[:, :-1], ((0, 0), (1, 0)), mode="constant")
    right = np.pad(mask[:, 1:], ((0, 0), (0, 1)), mode="constant")

    empty = mask == 0
    corner_fill = (
        (up & left) | (up & right) | (down & left) | (down & right)
    )
    tjunction_fill = (
        (left & right & (up | down)) | (up & down & (left | right))
    )
    fill_mask = (corner_fill | tjunction_fill) & empty
    result = mask.copy()
    result[fill_mask] = 1
    return result


def stitch(input_path: str, output_path: str) -> None:
    binary = load_binary_mask(input_path)
    hitl = flag_if_not_single_pixel(binary)
    segments = extract_segments(binary)

    classified = sum(1 for s in segments if s.orientation in {"horizontal", "vertical"})
    unclassified = [s for s in segments if s.orientation is None]
    if classified != len(segments):
        sys.stderr.write(
            f"[INFO] Segments classified: {classified}/{len(segments)}. "
            f"Unclassified: {len(unclassified)}.\n"
        )
        if unclassified:
            sys.stderr.write(
                "[INFO] Unclassified segments retained without modification.\n"
            )

    merged_lines, leftover = merge_axis_aligned(segments)

    stitched = draw_lines(binary.shape, merged_lines)

    # Preserve any unclassified pixels from the original skeleton
    for seg in leftover:
        for r, c in seg.path:
            stitched[r, c] = 1

    stitched = fill_orthogonal_gaps(stitched)

    cv2.imwrite(output_path, (stitched * 255).astype(np.uint8))

    if hitl:
        sys.stderr.write("[HITL] Manual review recommended before using stitched output.\n")
    print(f"Saved stitched skeleton to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitch fragmented wall skeletons")
    parser.add_argument("--input", required=True, help="Path to pruned skeleton image")
    parser.add_argument("--output", required=True, help="Path to write stitched skeleton")
    args = parser.parse_args()

    stitch(args.input, args.output)
