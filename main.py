import numpy as np
import cv2,glob, os, rootate_functions

path="C:/Users/simon/Desktop/Tesi/pyProject/"
ouput_path="C:/Users/simon/Desktop/Tesi/pyProject/rotated/"
list_of_files = os.listdir(path)
files=[x for x in list_of_files if x.endswith(".png") or x.endswith(".jpg")]

#for file in files:
    #rootate_functions.detect_optics(path=str(path+file), path_ooutput=ouput_path, file_name=file)
    #rootate_functions.canny_houge_rotation(img=str(path+file), ouput_path=ouput_path, file_name=file)

#
rootate_functions.find_sudoku(img=str(path+"9001.png"))