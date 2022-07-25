import numpy as np
import cv2
import math
import pytesseract
from scipy.ndimage import rotate as Rotate
from scipy import ndimage
import argparse

def detect_optics (path: str, path_ooutput: str, file_name: str):
    pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
    # Read image from which text needs to be extracted
    img = cv2.imread(path)

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

def canny_houge_rotation(img: str, ouput_path: str, file_name: str):
    img_before = cv2.imread(img)

    #cv2.namedWindow("Before", cv2.WINDOW_NORMAL)
    #cv2.imshow("Before", img_before)
    #key = cv2.waitKey(0)

    img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 200, apertureSize=5)

    #cv2.namedWindow("Canny", cv2.WINDOW_NORMAL)
    #cv2.imshow("Canny", img_edges)
    #key = cv2.waitKey(0)

    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

    angles = []

    for [[x1, y1, x2, y2]] in lines:
        cv2.line(img_before, (x1, y1), (x2, y2), (255, 0, 0), 2)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    #cv2.namedWindow("Detected lines", cv2.WINDOW_NORMAL)
    #cv2.imshow("Detected lines", img_before)
    #key = cv2.waitKey(0)
    median_angle = np.median(angles)
    if median_angle != float(-90.000):
        img_rotated = ndimage.rotate(img_before, median_angle)
        print(median_angle)
    else:
        img_rotated=img_before
        print("no angle")

    cv2.namedWindow("Image Rooteted", cv2.WINDOW_NORMAL)
    cv2.imshow("Image Rooteted", img_rotated)
    key = cv2.waitKey(0)

    cv2.imwrite(str(ouput_path+file_name), img_rotated)

def find_sudoku(img: str):
    image = cv2.imread(img)

    cv2.namedWindow("Before", cv2.WINDOW_NORMAL)
    cv2.imshow("Before", image)
    key = cv2.waitKey(0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
    cv2.imshow("gray", gray)
    key = cv2.waitKey(0)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    cv2.namedWindow("blur", cv2.WINDOW_NORMAL)
    cv2.imshow("blur", blur)
    key = cv2.waitKey(0)

    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    cv2.namedWindow("thresh", cv2.WINDOW_NORMAL)
    cv2.imshow("thresh", thresh)
    key = cv2.waitKey(0)

    contours,hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    c = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > max_area:
            max_area = area
            best_cnt = i
            image = cv2.drawContours(image, contours, c, (0, 255, 0), 3)
        c += 1

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", image)
    key = cv2.waitKey(0)

    mask = np.zeros((gray.shape), np.uint8)
    cv2.drawContours(mask, [best_cnt], 0, 255, -1)
    cv2.drawContours(mask, [best_cnt], 0, 0, 2)


    """
    out = np.zeros_like(gray)
    out[mask == 255] = gray[mask == 255]


    blur = cv2.GaussianBlur(out, (5, 5), 0)

    cv2.namedWindow("blur", cv2.WINDOW_NORMAL)
    cv2.imshow("blur", blur)
    key = cv2.waitKey(0)


    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    cv2.namedWindow("thresh", cv2.WINDOW_NORMAL)
    cv2.imshow("thresh", thresh)
    key = cv2.waitKey(0)

    contours,hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    c = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 1000 / 2:
            cv2.drawContours(image, contours, c, (0, 255, 0), 3)
        c += 1

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", image)
    key = cv2.waitKey(0)
    """
    cv2.destroyAllWindows()

