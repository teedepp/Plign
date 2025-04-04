import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm  # Progress bar

# Paths
BASE_FOLDER = "assignments_processed"
OUTPUT_CSV = "similar_images.csv"

def load_images():
    """Load all images from subfolders."""
    image_paths = {}
    for folder in os.listdir(BASE_FOLDER):
        folder_path = os.path.join(BASE_FOLDER, folder)
        if os.path.isdir(folder_path):
            image_paths[folder] = [
                os.path.join(folder_path, img_file)
                for img_file in os.listdir(folder_path)
                if img_file.endswith((".jpg", ".png"))
            ]
    return image_paths

def calculate_similarity(img1_path, img2_path):
    """Compute Structural Similarity Index (SSIM) between two images."""
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        return 0  # Return 0 if image loading fails

    # Resize images to the same dimensions
    img1 = cv2.resize(img1, (500, 700))
    img2 = cv2.resize(img2, (500, 700))

    score, _ = ssim(img1, img2, full=True)
    return round(score * 100, 2)  # Convert to percentage

def batch_process_images():
    """Compare images across different student folders, skipping unnecessary checks."""
    image_paths = load_images()
    results = []
    matched_folders = set()  # Keep track of which folders already have a match

    folder_list = list(image_paths.keys())
    total_comparisons = sum(len(image_paths[f1]) * sum(len(image_paths[f2]) for f2 in folder_list[i+1:]) for i, f1 in enumerate(folder_list))

    with tqdm(total=total_comparisons, desc="Processing Images", unit=" comparisons") as pbar:
        for i, folder1 in enumerate(folder_list):
            if folder1 in matched_folders:
                continue  # Skip if already matched
            
            for j in range(i + 1, len(folder_list)):  # Compare only with later folders
                folder2 = folder_list[j]
                if folder2 in matched_folders:
                    continue  # Skip if folder2 already has a match

                for img1_path in image_paths[folder1]:
                    for img2_path in image_paths[folder2]:
                        similarity = calculate_similarity(img1_path, img2_path)
                        pbar.update(1)  # Update progress bar

                        if similarity == 100.0:
                            results.append([img1_path, img2_path, similarity])
                            print(f"✅ 100% Match Found: {folder1} & {folder2}")
                            matched_folders.add(folder1)
                            matched_folders.add(folder2)
                            break  # Stop checking more images for this folder
                    if folder1 in matched_folders:
                        break  # Exit if already matched
                if folder1 in matched_folders:
                    break  # Exit if already matched

    # Save results to CSV
    df = pd.DataFrame(results, columns=["Image1", "Image2", "Similarity_Score"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Similar images saved to {OUTPUT_CSV}")

# Run the script
if __name__ == "__main__":
    batch_process_images()
