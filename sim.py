import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def are_images_identical(image1_path, image2_path, threshold=1.0):
    """Check if two images are 100% identical using SSIM"""
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Ensure images are of the same size
    if img1.shape != img2.shape:
        return False, 0.0  # Not identical

    # Compute SSIM (Structural Similarity Index)
    similarity, _ = ssim(img1, img2, full=True)

    return similarity >= threshold, similarity

# Example Usage
image1 = "hs4.jpg"
image2 = "hs4 copy.jpg"

is_identical, similarity_score = are_images_identical(image1, image2)

if is_identical:
    print(f"✅ Images are **100% identical** (SSIM: {similarity_score:.2f})")
else:
    print(f"❌ Images are **not identical** (SSIM: {similarity_score:.2f})")
