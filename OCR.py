
import pytesseract
import tempfile

from scipy import ndimage
import numpy as np

import math
import cv2

def detect_optics (img: np.ndarray, path_ooutput: str, file_name: str):
    pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
    # Read image from which text needs to be extracted

    # Preprocessing the image starts

    # Convert the image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Performing OTSU threshold
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # Specify structure shape and kernel size.
    # Kernel size increases or decreases the area
    # of the rectangle to be detected.
    # A smaller value like (10, 10) will detect
    # each word instead of a sentence.
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

    # Finding contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)

    # Creating a copy of image
    im2 = img.copy()

    # A text file is created and flushed
    file = open(str(path_ooutput+file_name+"_recognized.txt"), "w+")
    file.write("")
    file.close()

    # Looping through the identified contours
    # Then rectangular part is cropped and passed on
    # to pytesseract for extracting text from it
    # Extracted text is then written into the text file
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Drawing a rectangle on copied image
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Cropping the text block for giving input to OCR
        cropped = im2[y:y + h, x:x + w]

        # Open the file in append mode
        file = open(str(path_ooutput+file_name+"_recognized.txt"), "a")

        # Apply OCR on the cropped image
        text = pytesseract.image_to_string(cropped)

        # Appending the text into file
        file.write(text)
        file.write("\n")

        # Close the file
        file.close
    return str(path_ooutput+file_name+"_recognized.txt")

def get_info_patient (file: str):
    with open(file) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            line = fp.readline()
            if "ID:" in line:
                id=line[3:]
                return id
