import cv2
import pytesseract
import numpy as np
import fitz



def ocr_image(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow("RGB Image", grayscale)

    # Wait for a key event and close the window when any key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    text = pytesseract.image_to_string(grayscale)
    return text


def process_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    print('Read PDF')

    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        image_matrix = page.get_pixmap()
        

        # Convert image to numpy array and BGR format (OpenCV's standard format)
        image = np.frombuffer(image_matrix.samples, dtype=np.uint8).reshape(
            image_matrix.h, image_matrix.w, 3
        )

        ocr_text = ocr_image(image)
        
 

if __name__ == "__main__":
    pdf_path = "Jewelosco.pdf"
    process_pdf(pdf_path)