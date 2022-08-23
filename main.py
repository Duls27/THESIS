import pathlib
import time

import numpy as np
import cv2, os, functions, tempfile, OCR

path="C:/Users/simon/Desktop/Tesi/pyProject/ECGs/"
outpath="C:/Users/simon/Desktop/Tesi/pyProject/optics/"
outpath2="C:/Users/simon/Desktop/Tesi/pyProject/grid removed/"
outpath3="C:/Users/simon/Desktop/Tesi/pyProject/cca/"


list_of_files = os.listdir(path)
files=[x for x in list_of_files if x.endswith(".png") or x.endswith(".jpg")]

temp_dir = tempfile.TemporaryDirectory()

for file in files:
    start=time.perf_counter()
    print (f"\nProcessing file: {file}")
    path_to_img=str(path+str(file))

    img=cv2.imread(path_to_img)

    print(f"\t Rotating...")
    img_rooteted=functions.canny_houge_rotation(img)

    print(f"\t Cropping...")
    ecg, subject_data=functions.crop_ecg_vs_optics(img_rooteted)
    ecg_tmp_path=os.path.join(temp_dir.name, "/", file)

    cv2.imwrite(ecg_tmp_path, ecg)

    cv2.destroyAllWindows()

    print(f"\t Removing grid...")
    cropped=functions.crop_biggest_rect(img=ecg_tmp_path, path_ooutput=outpath2, file_name=file)
    functions.cca_analisys(img=cropped, file_name=file, path_output=outpath3)

    print(f"\t Detecting opticts...")
    OCR.detect_optics(img=subject_data, path_ooutput=outpath, file_name=file)

    stop=time.perf_counter()
    print(f"File {file} processed in: {stop-start}")

    temp_dir.cleanup()

print("Process ENDED!")