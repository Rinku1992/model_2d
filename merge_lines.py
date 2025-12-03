# preprocessing.py
import cv2
import numpy as np

def load_image(input_path):
    """
    Load the input image as a grayscale image.
    """
    return cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

def preprocess_image(image):
    """
    Binarize the input image (skeleton).
    """
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return binary_image

# skeletonize_walls.py
import cv2
import numpy as np

def skeletonize_image(image):
    """
    Perform morphological thinning to reduce the walls to 1-pixel thick skeleton.
    """
    return cv2.ximgproc.thinning(image)

def remove_small_components(skeleton, min_size=100):
    """
    Removes small components in the skeleton to avoid noise.
    """
    # Find contours and remove small ones
    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < min_size:
            cv2.drawContours(skeleton, [contour], -1, (0, 0, 0), -1)
    return skeleton

def skeletonize_and_prune(image, min_size=100):
    """
    Skeletonize the image and remove small components.
    """
    skeleton = skeletonize_image(image)
    pruned = remove_small_components(skeleton, min_size)
    return pruned

# vectorize_lines.py
import numpy as np

def classify_lines(segments):
    """
    Classify lines into vertical and horizontal.
    """
    horizontal, vertical = [], []
    for seg in segments:
        x1, y1, x2, y2 = seg
        if abs(y2 - y1) < abs(x2 - x1):
            horizontal.append(seg)
        else:
            vertical.append(seg)
    return horizontal, vertical

def merge_lines(segments, axis='x', threshold=2):
    """
    Merge lines that are jagged but close in proximity along the given axis.
    """
    merged = []
    segments.sort(key=lambda s: s[axis])  # Sort by the axis (x or y)
    current_line = list(segments[0])  # Start with the first line
    for i in range(1, len(segments)):
        if abs(segments[i][axis] - current_line[axis]) <= threshold:
            current_line[axis] = max(current_line[axis], segments[i][axis])  # Merge the lines
        else:
            merged.append(tuple(current_line))  # Add merged line
            current_line = list(segments[i])  # Start a new line
    merged.append(tuple(current_line))  # Add last line
    return merged

def vectorize_skeleton(image, threshold=2):
    """
    Convert skeleton image to lines and merge jagged lines.
    """
    # Find contours (lines) from the skeleton image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours:
        for i in range(0, len(contour) - 1, 2):
            pt1, pt2 = contour[i][0], contour[i+1][0]
            lines.append([pt1[0], pt1[1], pt2[0], pt2[1]])

    # Classify and merge lines
    horizontal, vertical = classify_lines(lines)
    horizontal = merge_lines(horizontal, axis='x', threshold=threshold)
    vertical = merge_lines(vertical, axis='y', threshold=threshold)

    return horizontal + vertical

# pipeline.py
import cv2
import numpy as np
from preprocessing import load_image, preprocess_image
from skeletonize_walls import skeletonize_and_prune
from vectorize_lines import vectorize_skeleton

def run_pipeline(input_path, output_path, min_line_length=5, gap_threshold=2):
    """
    Orchestrates the full pipeline from reading image to generating output.
    """
    # Step 1: Load and preprocess image
    image = load_image(input_path)
    binary_image = preprocess_image(image)

    # Step 2: Skeletonize and prune the image
    pruned_skeleton = skeletonize_and_prune(binary_image)

    # Step 3: Vectorize the pruned skeleton
    lines = vectorize_skeleton(pruned_skeleton, threshold=gap_threshold)

    # Step 4: Draw and save the result
    output_image = np.ones_like(binary_image) * 255  # Start with a white image
    for line in lines:
        cv2.line(output_image, (line[0], line[1]), (line[2], line[3]), (0, 0, 0), 1)

    cv2.imwrite(output_path, output_image)
    print(f"Pipeline complete. Output saved to {output_path}")

# Command line usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Floorplan cleaning pipeline")
    parser.add_argument('--input', type=str, required=True, help="Input image path")
    parser.add_argument('--output', type=str, required=True, help="Output image path")
    parser.add_argument('--min-line-length', type=int, default=5, help="Minimum line length")
    parser.add_argument('--gap-threshold', type=int, default=2, help="Maximum gap for merging lines")

    args = parser.parse_args()
    run_pipeline(args.input, args.output, args.min_line_length, args.gap_threshold)
