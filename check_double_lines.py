import cv2
import numpy as np

def detect_double_lines(image_path: str, output_path: str) -> None:
    """
    Detect double lines in a skeletonized image by comparing original and eroded images.

    Parameters:
    ----------
    image_path : str
        Path to the skeletonized image (uint8 {0,255}).
    output_path : str
        Path where the output image with detected double lines will be saved.
    """
    # Load the skeletonized image (assumed to be in binary form)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    # Perform morphological erosion to shrink the lines
    kernel = np.ones((3, 3), np.uint8)  # A 3x3 square kernel
    eroded_img = cv2.erode(img, kernel, iterations=1)  # Erosion reduces line thickness by 1 pixel

    # Calculate the difference between the original and eroded images
    diff_img = cv2.subtract(img, eroded_img)

    # Mark areas of double lines as white (255) in the output image
    double_lines_img = np.zeros_like(img)
    double_lines_img[diff_img > 0] = 255  # Areas where difference > 0 are marked as double lines

    # Save the output image where double lines are detected
    cv2.imwrite(output_path, double_lines_img)

    print(f"Double lines detection complete. Output saved at: {output_path}")


if __name__ == "__main__":
    # Input and output paths
    input_image = "output/04_skeleton_pruned.png"  # Path to the pruned skeleton image
    output_image = "output/detected_double_lines.png"  # Path to save the result
    
    detect_double_lines(input_image, output_image)
