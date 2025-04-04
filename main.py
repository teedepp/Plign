# Import Libraries
from paddleocr import PaddleOCR
import cv2
import numpy as np
import requests
import json

# OpenRouter API Key
API_KEY = "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"

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# Function: Preprocess Image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    img = cv2.GaussianBlur(img, (5,5), 0)  # Reduce noise
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)  # Enhance text contrast
    return img

# Function: Perform OCR - Preserving Structure
def extract_text(image_path):
    result = ocr.ocr(image_path, cls=True)
    
    # Preserve the line structure
    structured_text = []
    for line in result:
        line_text = []
        for word in line:
            line_text.append(word[1][0])  # Extract recognized text
        structured_text.append(" ".join(line_text))  # Join words in each line
    
    # Return both structured and flat versions
    return {
        "structured": structured_text,  # List of lines
        "flat": " ".join(structured_text)  # Single string for LLM processing
    }

# Function: Correct Text using DeepSeek R1
def correct_text_with_deepseek(ocr_text):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    prompt = """
    There is some ambiguity and spelling mistakes in the following text. 
    Correct the spelling mistakes while keeping the context in mind. 
    Preserve the original line structure by separating lines with newline characters.
    Just provide the corrected text without additional comments:

    {text}
    """

    payload = {
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "messages": [
            {
                "role": "user",
                "content": prompt.format(text=ocr_text)
            }
        ],
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print("Error:", response.json())
        return None

# Load and Preprocess Image
image_path = "hs4.jpg"
preprocessed_img = preprocess_image(image_path)
cv2.imwrite("preprocessed.jpg", preprocessed_img)

# Perform OCR
ocr_result = extract_text(image_path)
structured_text = ocr_result["structured"]
flat_text = ocr_result["flat"]

# Apply DeepSeek R1 for Text Correction
corrected_text = correct_text_with_deepseek(flat_text)

# Print Results
print("\nRaw Extracted Text (Structured):")
for line in structured_text:
    print(line)

print("\nRaw Extracted Text (Flat for LLM):")
print(flat_text)

print("\nCorrected Text:")
print(corrected_text)