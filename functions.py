import pytesseract
from scipy import ndimage
import numpy as np
from PIL import Image
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

def canny_houge_rotation(img_before: np.ndarray):
    img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 200, apertureSize=5)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

    angles = []
    for [[x1, y1, x2, y2]] in lines:
        #cv2.line(img_before, (x1, y1), (x2, y2), (255, 0, 0), 2)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    median_angle = np.median(angles)
    if median_angle != float(-90.000):
        img_rotated = ndimage.rotate(img_before, median_angle)
    else:
        img_rotated=img_before

    cv2.namedWindow("Image Rooteted", cv2.WINDOW_NORMAL)
    cv2.imshow("Image Rooteted", img_rotated)
    key = cv2.waitKey(0)

    return img_rotated

def crop_ecg_optics (img: np.ndarray):
    pic=img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    c = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > max_area:
            max_area = area
            best_cnt = i
            #img = cv2.drawContours(img, contours, c, (0, 255, 0), 3)
        c += 1

    mask = np.zeros((gray.shape), np.uint8)
    cv2.drawContours(mask, [best_cnt], 0, 255, -1)

    out = np.zeros_like(gray)
    out[mask == 255] = gray[mask == 255]
    ones_part = out[~np.all(out == 0, axis=1)]
    zero_part = out[np.all(out == 0, axis=1)]

    ecg= pic[pic.shape[0]-ones_part.shape[0]:, pic.shape[1]-ones_part.shape[1]:]
    subj_data=pic[:zero_part.shape[0], :zero_part.shape[1]]

    cv2.namedWindow("ecg", cv2.WINDOW_NORMAL)
    cv2.imshow("ecg", ecg)
    key = cv2.waitKey(0)

    cv2.namedWindow("subj_data", cv2.WINDOW_NORMAL)
    cv2.imshow("subj_data", subj_data)
    key = cv2.waitKey(0)

    return ecg, subj_data






def find_sudoku(img: str):
    image = cv2.imread(img)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
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
    #cv2.drawContours(mask, [best_cnt], 0, 0, 2)

    out = np.zeros_like(gray)
    out[mask == 255] = gray[mask == 255]


    blur = cv2.GaussianBlur(out, (5, 5), 0)

    cv2.namedWindow("blurred_mask", cv2.WINDOW_NORMAL)
    cv2.imshow("blurred_mask", out)
    key = cv2.waitKey(0)
    """

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

def cca_analisys (img: str):
    # Loading the image
    img = cv2.imread(img)

    # preprocess the image
    gray_img = cv2.cvtColor(img,
                            cv2.COLOR_BGR2GRAY)

    # Applying 7x7 Gaussian Blur
    blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)

    # Applying threshold
    threshold = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Apply the Component analysis function
    analysis = cv2.connectedComponentsWithStats(threshold,
                                                4,
                                                cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis

    # Initialize a new image to
    # store all the output components
    output = np.zeros(gray_img.shape, dtype="uint8")

    # Loop through each component
    for i in range(1, totalLabels):

        # Area of the component
        area = values[i, cv2.CC_STAT_AREA]

        if (area > 140) and (area < 400):
            # Create a new image for bounding boxes
            new_img = img.copy()

            # Now extract the coordinate points
            x1 = values[i, cv2.CC_STAT_LEFT]
            y1 = values[i, cv2.CC_STAT_TOP]
            w = values[i, cv2.CC_STAT_WIDTH]
            h = values[i, cv2.CC_STAT_HEIGHT]

            # Coordinate of the bounding box
            pt1 = (x1, y1)
            pt2 = (x1 + w, y1 + h)
            (X, Y) = centroid[i]

            # Bounding boxes for each component
            cv2.rectangle(new_img, pt1, pt2,
                          (0, 255, 0), 3)
            cv2.circle(new_img, (int(X),
                                 int(Y)),
                       4, (0, 0, 255), -1)

            # Create a new array to show individual component
            component = np.zeros(gray_img.shape, dtype="uint8")
            componentMask = (label_ids == i).astype("uint8") * 255

            # Apply the mask using the bitwise operator
            component = cv2.bitwise_or(component, componentMask)
            output = cv2.bitwise_or(output, componentMask)

            # Show the final images
            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.imshow("image", new_img)
            key = cv2.waitKey(0)

            cv2.namedWindow("Individual Component", cv2.WINDOW_NORMAL)
            cv2.imshow("Individual Component", component)

            cv2.namedWindow("Filtered Components", cv2.WINDOW_NORMAL)
            cv2.imshow("Filtered Components", output)

    cv2.destroyAllWindows()

