# skeletonize_walls.py
"""
Skeletonization + safe pruning.

Pipeline:
1. Take the binary wall mask (uint8 {0,255}).
2. Skeletonize to 1-pixel-wide ridges using skimage.morphology.skeletonize.
3. Remove ONLY tiny isolated components (noise), NOT endpoints of long walls.

Why this change?
----------------
The previous version pruned "spurs" by removing every endpoint (pixels with only
1 neighbor) on every iteration. On a straight wall, the endpoints are real
geometry – so repeated pruning eventually deleted the whole wall.

Here we switch to a safer strategy:
- Keep the skeleton as-is.
- Compute connected components.
- Drop only components with very few pixels (small blobs).
This preserves all long wall skeletons while cleaning small noise.

Key knob:
- spur_iterations: interpreted as how aggressively we remove small components.
  min_keep_pixels = 2 + spur_iterations * 10
  - 0 → keep almost everything (remove only 1-pixel specks)
  - 1 → remove components with < 12 pixels
  - 2 → remove components with < 22 pixels
"""

from __future__ import annotations
import os
from typing import Tuple, Optional

import cv2
import numpy as np
from skimage.morphology import skeletonize


def _binary_to_bool(img: np.ndarray) -> np.ndarray:
    """Convert uint8 {0,255} to boolean mask."""
    return img > 0


def _bool_to_binary(img_bool: np.ndarray) -> np.ndarray:
    """Convert boolean mask back to uint8 {0,255}."""
    return (img_bool.astype(np.uint8) * 255)


def _remove_small_components(skel: np.ndarray, min_keep_pixels: int) -> np.ndarray:
    """
    Remove connected components smaller than `min_keep_pixels`.

    skel           : uint8 {0,255} skeleton
    min_keep_pixels: components with pixel count < min_keep_pixels are dropped

    This is a SAFE way to prune:
    - very small blobs (noise) disappear
    - long walls (hundreds of pixels) are preserved completely
    """
    # Work on a 0/1 mask
    mask = (skel > 0).astype(np.uint8)

    # Connected components (8-connectivity)
    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)

    # labels == 0 is background
    cleaned = np.zeros_like(mask, dtype=np.uint8)

    for label in range(1, num_labels):
        component = (labels == label)
        count = int(component.sum())
        if count >= min_keep_pixels:
            cleaned[component] = 1  # keep this component

    return (cleaned * 255).astype(np.uint8)


def skeletonize_and_prune(
    binary: np.ndarray,
    spur_iterations: int = 1,
    debug_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Skeletonize + SAFE prune.

    Parameters
    ----------
    binary : np.ndarray
        Input wall mask, uint8 {0,255}, where 255 = wall.
    spur_iterations : int, optional
        Controls how aggressively small components are removed.
        We map this to a minimum component size:

            min_keep_pixels = 2 + spur_iterations * 10

        Examples:
        - 0 → min_keep_pixels = 2   (removes only 1-pixel specks)
        - 1 → min_keep_pixels = 12  (removes tiny blobs)
        - 2 → min_keep_pixels = 22  (slightly more aggressive)

        IMPORTANT:
        This does *not* delete endpoints of long walls, only small blobs.
    debug_dir : str or None
        If provided, intermediate images are written there:
        - 03_skeleton.png
        - 04_skeleton_pruned.png

    Returns
    -------
    skel : np.ndarray
        Raw 1-pixel skeleton (uint8 {0,255}).
    skel_pruned : np.ndarray
        Skeleton after safe pruning of small components.
    """
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    # 1) Normalize to bool
    bool_img = _binary_to_bool(binary)

    # 2) Skeletonize → boolean skeleton
    skel_bool = skeletonize(bool_img)
    skel = _bool_to_binary(skel_bool)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "03_skeleton.png"), skel)

    # 3) SAFE pruning: remove only tiny components
    #    Map spur_iterations -> min_keep_pixels threshold
    min_keep_pixels = 2 + max(0, spur_iterations) * 10
    skel_pruned = _remove_small_components(skel, min_keep_pixels=min_keep_pixels)

    # NOTE:
    # We deliberately do NOT erode the skeleton further here.
    # skimage.skeletonize already produces 1-pixel wide lines. Extra erosion
    # risks deleting thin segments or breaking connectivity.

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "04_skeleton_pruned.png"), skel_pruned)

    return skel, skel_pruned


# Optional: quick standalone test
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Test skeletonization + safe pruning")
    parser.add_argument("--input", required=True, help="Binary wall image (white walls on black)")
    parser.add_argument("--output-dir", default="debug_skeleton", help="Where to save debug PNGs")
    parser.add_argument("--spur-iterations", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read input image: {args.input}")

    # Assume input already binary-ish; threshold to be safe
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    skel, skel_pruned = skeletonize_and_prune(
        binary,
        spur_iterations=args.spur_iterations,
        debug_dir=args.output_dir,
    )

    cv2.imwrite(os.path.join(args.output_dir, "01_input_binary.png"), binary)
    cv2.imwrite(os.path.join(args.output_dir, "03_skeleton.png"), skel)
    cv2.imwrite(os.path.join(args.output_dir, "04_skeleton_pruned.png"), skel_pruned)
    print("Done. Check images in:", args.output_dir)
