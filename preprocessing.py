import os
from typing import Optional

import cv2
import numpy as np


def load_grayscale(path: str) -> np.ndarray:
    """Load image as grayscale. Raises if not found."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def preprocess_floorplan(
    gray: np.ndarray,
    debug_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Convert grayscale floorplan to a clean binary mask (uint8, {0,255}).

    Walls become white (255), background black (0).
    """
    # Create debug directory if it doesn't exist
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    # 1) Slight blur to remove salt-and-pepper, but keep edges sharp.
    BLUR_KSIZE = 5  # you can try 3 or 7 later
    blurred = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)

    # 2) Adaptive threshold → robust to lighting / background gradients.
    ADAPTIVE_BLOCK = 51  # window size
    ADAPTIVE_C = 10
    bin_inv = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK,
        ADAPTIVE_C,
    )
    # binary: foreground=255, background=0
    binary = bin_inv

    # 3) Morphological closing to seal tiny gaps in walls.
    MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    MORPH_ITERS = 1
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=MORPH_ITERS)

    # Save the binary output to the 'output' folder
    output_path = os.path.join("output", "binary_output.png")
    cv2.imwrite(output_path, closed)
    print(f"Binary image saved at: {output_path}")

    # Optionally save debug images if debug_dir is provided
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "01_gray.png"), gray)
        cv2.imwrite(os.path.join(debug_dir, "02_binary.png"), closed)

    return closed


if __name__ == "__main__":
    input_path = "input/image (1).png"  # Update this path to your actual input file path
    debug_dir = "output/debug"  # Folder where debug images will be saved (optional)

    # Ensure the output folder exists
    os.makedirs("output", exist_ok=True)

    # Load the image and process
    gray = load_grayscale(input_path)
    binary = preprocess_floorplan(gray, debug_dir)
