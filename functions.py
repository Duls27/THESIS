from scipy import ndimage
import numpy as np
import math
import cv2


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

    '''
    cv2.namedWindow("Image Rooteted", cv2.WINDOW_NORMAL)
    cv2.imshow("Image Rooteted", img_rotated)
    key = cv2.waitKey(0)
    '''
    return img_rotated

def crop_ecg_vs_optics (img: np.ndarray):
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

    '''
    cv2.namedWindow("ecg", cv2.WINDOW_NORMAL)
    cv2.imshow("ecg", ecg)
    key = cv2.waitKey(0)

    cv2.namedWindow("subj_data", cv2.WINDOW_NORMAL)
    cv2.imshow("subj_data", subj_data)
    key = cv2.waitKey(0)
    '''

    return ecg, subj_data

def cca_analisys (img: str, file_name, path_output:str):
    # Loading the image
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

            '''
            # Show the final images
            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.imshow("image", new_img)
            key = cv2.waitKey(0)

            cv2.namedWindow("Individual Component", cv2.WINDOW_NORMAL)
            cv2.imshow("Individual Component", component)

            cv2.namedWindow("Filtered Components", cv2.WINDOW_NORMAL)
            cv2.imshow("Filtered Components", output)
            '''


    cv2.destroyAllWindows()
    cv2.imwrite(str(path_output+file_name), output)


def crop_biggest_rect (img: str, path_ooutput:str, file_name:str):


    image = cv2.imread(img)

    #riquadra nuovamente l'ecg togliendo i borfi bianchi

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
            # img = cv2.drawContours(img, contours, c, (0, 255, 0), 3)
        c += 1

    mask = np.zeros((gray.shape), np.uint8)
    cv2.drawContours(mask, [best_cnt], 0, 1, -1)

    y_sum=np.sum(mask, axis=1)
    y_nz=np.nonzero(y_sum)
    y=y_nz[0][0]
    h_end=y_nz[0][len(y_nz[0])-1]
    h_start=y_nz[0][0]
    h= h_end-h_start
    x_sum=np.sum(mask, axis=0)
    x_nz=np.nonzero(x_sum)
    x = x_nz[0][0]
    w_end=x_nz[0][len(x_nz[0]) - 1]
    w_start=x_nz[0][0]
    w = w_end-w_start

    crop = image[y:y + h, x:x + w]

    '''
    # Conversion to CMYK (just the K channel):

    # Convert to float and divide by 255:
    imgFloat = img.astype(np.float) / 255.

    # Calculate channel K:
    kChannel = 1 - np.max(imgFloat, axis=2)

    # Convert back to uint 8:
    kChannel = (255 * kChannel).astype(np.uint8)

    # Threshold image:
    binaryThresh = 190
    _, binaryImage = cv2.threshold(kChannel, binaryThresh, 255, cv2.THRESH_BINARY)

    # Filter small blobs:
    minArea = 100
    #binaryImage = areaFilter(minArea, binaryImage)

    # Use a little bit of morphology to clean the mask:
    # Set kernel (structuring element) size:
    kernelSize = 3
    # Set morph operation iterations:
    opIterations = 2
    # Get the structuring element:
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    # Perform closing:
    binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, morphKernel, None, None, opIterations,
                                   cv2.BORDER_REFLECT101)

    invert = cv2.bitwise_not(binaryImage)
    '''
    # save image
    cv2.imwrite((path_ooutput+file_name), crop)
    return crop
