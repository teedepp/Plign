import os
from pdf2image import convert_from_path

# Paths
ASSIGNMENTS_FOLDER = "submissions"
EVAL_FOLDER = "assignments_processed"

def convert_pdf_to_images(pdf_path, output_folder, folder_name):
    """Convert PDF pages to images and store in output folder with formatted names."""
    images = convert_from_path(pdf_path)
    for i, img in enumerate(images):
        img_name = f"{folder_name}_page_{i+1}.jpg"  # New naming format
        img_path = os.path.join(output_folder, img_name)
        img.save(img_path, "JPEG")
        print(f"✅ Saved {img_path}")

def process_assignments():
    """Create subfolders in 'assg_eval' and store converted images."""
    os.makedirs(EVAL_FOLDER, exist_ok=True)
    
    for pdf_file in os.listdir(ASSIGNMENTS_FOLDER):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(ASSIGNMENTS_FOLDER, pdf_file)
            folder_name = os.path.splitext(pdf_file)[0]  # Extract name without extension
            student_folder = os.path.join(EVAL_FOLDER, folder_name)

            os.makedirs(student_folder, exist_ok=True)  # Create student subfolder
            print(f"📂 Created folder: {student_folder}")

            convert_pdf_to_images(pdf_path, student_folder, folder_name)  # Convert & save images

# Run the script
if __name__ == "__main__":
    process_assignments()
