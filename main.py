import numpy as np
import cv2,glob, os, functions

path="C:/Users/simon/Desktop/Tesi/pyProject/"
outpath="C:/Users/simon/Desktop/Tesi/pyProject/optics/"
list_of_files = os.listdir(path)
files=[x for x in list_of_files if x.endswith(".png") or x.endswith(".jpg")]

for file in files:
    path_to_img=str(path+str(file))

    img=cv2.imread(path_to_img)

    img_rooteted=functions.canny_houge_rotation(img)

    ecg, subject_data=functions.crop_ecg_optics(img_rooteted)

    cv2.destroyAllWindows()

    functions.detect_optics(img=subject_data, path_ooutput=outpath, file_name=file)

    break