from fastapi import FastAPI, UploadFile, HTTPException
import fitz  # PyMuPDF
from pyzbar.pyzbar import decode
from PIL import Image, ImageEnhance
import io
import cv2
import numpy as np
import requests

app = FastAPI()

# Function to extract images from PDF
def extract_images_from_pdf(pdf_file):
    images = []
    pdf_document = fitz.open(stream=pdf_file, filetype="pdf")

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        image_list = page.get_images(full=True)

        for img in image_list:
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]

            # Open image with PIL for further processing
            image = Image.open(io.BytesIO(image_bytes))
            # Resize image for better resolution
            image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
            images.append(image)

    return images

# Enhanced preprocessing function
def enhanced_preprocess_image(image):
    # Convert image to grayscale
    gray_image = np.array(image.convert("L"))
    
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Adaptive thresholding
    thresholded_image = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Normalize the image
    normalized_image = cv2.normalize(thresholded_image, None, 0, 255, cv2.NORM_MINMAX)
    
    return normalized_image

# Function to detect and extract QR code text using pyzbar
def extract_qr_from_images_pyzbar(images):
    qr_texts = []
    for image in images:
        preprocessed_image = enhanced_preprocess_image(image)
        qr_codes = decode(preprocessed_image)

        for qr in qr_codes:
            qr_texts.append(qr.data.decode("utf-8"))
    
    return qr_texts

# Function to detect and extract QR code using OpenCV
def extract_qr_from_images_opencv(images):
    qr_texts = []
    detector = cv2.QRCodeDetector()

    for image in images:
        # Convert to OpenCV-compatible format
        open_cv_image = np.array(image)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        # Try detecting and decoding QR codes
        data, bbox, _ = detector.detectAndDecode(open_cv_image)
        if bbox is not None and data:
            qr_texts.append(data)
    
    return qr_texts

# Function to send extracted QR code to API
def send_to_api(qr_text):
    url = "http://200.healthandglow.in/hgsupertax/api/GenerateQR/detailnew"
    payload = {
        "poNumber": "6090851",
        "vendorCode": "test1",
        "qrtext": qr_text
    }
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return {"status": "success", "response": response.json()}
        else:
            return {"status": "failed", "code": response.status_code, "message": response.text}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}

# FastAPI endpoint to upload a PDF and process it
@app.post("/extract-qrcode/")
async def extract_qr_code_from_pdf(file: UploadFile):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File format not supported. Please upload a PDF.")

    try:
        pdf_content = await file.read()
        images = extract_images_from_pdf(pdf_content)

        if not images:
            return {"status": "failed", "message": "No images found in the PDF."}

        # Try using both Pyzbar and OpenCV for better results
        qr_texts_pyzbar = extract_qr_from_images_pyzbar(images)
        qr_texts_opencv = extract_qr_from_images_opencv(images)

        # Combine the results from both methods
        qr_texts = list(set(qr_texts_pyzbar + qr_texts_opencv))  # Remove duplicates

        if qr_texts:
            results = []
            for qr_text in qr_texts:
                api_response = send_to_api(qr_text)
                results.append({"qr_text": qr_text, "api_response": api_response})
            return {"status": "success", "results": results}
        else:
            return {"status": "failed", "message": "No QR codes detected."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
