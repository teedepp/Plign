import os
import cv2
import json
import requests
from paddleocr import PaddleOCR

# Setup
API_KEY = "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"
ocr = PaddleOCR(use_angle_cls=True, lang="en")

input_folder = "assignments_processed"
output_folder = "assignments_texts"

# Preprocessing
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    return img

# OCR
def extract_text(image_path):
    result = ocr.ocr(image_path, cls=True)
    extracted_text = []

    if result is None:
        print(f"[Warning] No OCR result for: {image_path}")
        return ""

    for line in result:
        if line is None:
            continue
        for word in line:
            if word is not None:
                extracted_text.append(word[1][0])

    return "\n".join(extracted_text)


# LLM Correction
def correct_text_with_deepseek(ocr_text):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "meta-llama/llama-3.2-3b-instruct:free",
        "messages": [
            {
                "role": "user",
                "content": f"There is some ambiguity and spelling mistake in the following text. Correct the spelling mistakes whilst keeping the context in mind. I just want you to give the corrected text. Do not proceed with anything else:\n\n{ocr_text}"
            }
        ],
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print("Error from DeepSeek:", response.text)
        return ocr_text

# Batch Processing
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            input_path = os.path.join(root, file)
            rel_path = os.path.relpath(input_path, input_folder)
            output_path = os.path.join(output_folder, os.path.splitext(rel_path)[0] + ".txt")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            print(f"[INFO] Processing: {input_path}")
            try:
                pre_img = preprocess_image(input_path)
                temp_img_path = "temp.jpg"
                cv2.imwrite(temp_img_path, pre_img)

                raw_text = extract_text(temp_img_path)
                corrected_text = correct_text_with_deepseek(raw_text)

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(corrected_text)
                print(f"Saved: {output_path}")
            except Exception as e:
                print(f"[ERROR] Failed to process {input_path}: {e}")

