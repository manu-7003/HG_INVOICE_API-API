import fitz  # PyMuPDF
from pyzbar.pyzbar import decode
from PIL import Image, ImageEnhance
import io
import cv2
import numpy as np
import requests

# Function to extract images from PDF
def extract_images_from_pdf(pdf_path):
    images = []
    pdf_document = fitz.open(pdf_path)

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        image_list = page.get_images(full=True)

        for img in image_list:
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]

            # Open image with PIL for further processing
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)

    return images

# Function to preprocess image to improve QR code detection
def preprocess_image(image):
    # Convert the image to grayscale
    image = image.convert("L")

    # Increase contrast to make the QR code more visible
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)  # Enhance contrast

    # Convert back to an OpenCV-compatible format
    open_cv_image = np.array(image)
    return open_cv_image

# Function to detect and extract QR code text from images using pyzbar
def extract_qr_from_images_pyzbar(images):
    qr_texts = []
    for image in images:
        preprocessed_image = preprocess_image(image)
        qr_codes = decode(preprocessed_image)

        for qr in qr_codes:
            qr_texts.append(qr.data.decode("utf-8"))
    
    return qr_texts

# Function to detect and extract QR code using OpenCV
def extract_qr_from_images_opencv(images):
    qr_texts = []
    detector = cv2.QRCodeDetector()

    for image in images:
        open_cv_image = np.array(image)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        data, bbox, _ = detector.detectAndDecode(open_cv_image)
        if bbox is not None and data:
            qr_texts.append(data)
    
    return qr_texts

# Function to send extracted QR code to API
def send_to_api(qr_text):
    url = "http://200.healthandglow.in/hgsupertax/api/GenerateQR/detailnew"
    payload = {
        "poNumber": "1234",
        "vendorCode": "test1",
        "qrtext": qr_text
    }
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            print("Connection established successfully!")
            print("API Response:", response.json())  # Or response.text if not JSON
        else:
            print(f"Connection failed with status code: {response.status_code}")
            print("Response message:", response.text)
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to the API: {e}")

# Main function to handle the process
def extract_qr_from_pdf_and_send(pdf_path):
    images = extract_images_from_pdf(pdf_path)

    if not images:
        print("No images were extracted from the PDF.")
        return

    # Try using both Pyzbar and OpenCV for better results
    qr_texts_pyzbar = extract_qr_from_images_pyzbar(images)
    qr_texts_opencv = extract_qr_from_images_opencv(images)

    # Combine the results from both methods
    qr_texts = list(set(qr_texts_pyzbar + qr_texts_opencv))  # Remove duplicates

    if qr_texts:
        for qr_text in qr_texts:
            print(f"QR Code Found: {qr_text}")
            send_to_api(qr_text)  # Send each QR code to the API
    else:
        print("QR Code Not Found")

# Example usage
pdf_path = "BNA131_164_6047047.pdf"
extract_qr_from_pdf_and_send(pdf_path)
