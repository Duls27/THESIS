import cv2, os, functions, tempfile, OCR, time, win32com
from win32com.client import gencache
import win32com.client

path="C:/Users/simon/Desktop/Tesi/pyProject/ECGs/"
outpath="C:/Users/simon/Desktop/Tesi/pyProject/optics/"
outpath1="C:/Users/simon/Desktop/Tesi/pyProject/rootated/"
outpath2="C:/Users/simon/Desktop/Tesi/pyProject/grid removed/"
outpath3="C:/Users/simon/Desktop/Tesi/pyProject/RGBImages/"
outpath4="C:/Users/simon/Desktop/Tesi/pyProject/digitized/"


#Preprocessing Images
list_of_files = os.listdir(path)
files=[x for x in list_of_files if x.endswith(".png") or x.endswith(".jpg")]

temp_dir = tempfile.TemporaryDirectory()

for count,file in enumerate(files):
    start=time.perf_counter()
    print (f"\nProcessing file: {file}, \n\t{count+1}/{len(files)}")
    path_to_img=str(path+str(file))

    img=cv2.imread(path_to_img)

    print(f"\t Rotating...")
    img_rooteted=functions.canny_houge_rotation(img)
    cv2.imwrite(str(outpath1 + "canny" +file), img_rooteted)
    """
    fixme: np.ndarray = cv2.imread(path_to_img)
    cv2.imwrite(str(outpath1+file), functions.align_image(fixme))
    """

    print(f"\t Cropping...")
    ecg, subject_data=functions.crop_ecg_vs_optics(img_rooteted)

    print(f"\t Detecting opticts...")
    file_ocr=OCR.detect_optics(img=subject_data, path_ooutput=outpath, file_name=file)
    id=OCR.get_info_patient(file=file_ocr)

    ecg_tmp_path=os.path.join(temp_dir.name, "/", file)
    cv2.imwrite(ecg_tmp_path, ecg)

    cv2.destroyAllWindows()

    print(f"\t Removing grid...")

    cropped=functions.crop_biggest_rect(img=ecg_tmp_path, path_ooutput=outpath2, file_name=file)
    functions.remove_grid(img=cropped, file_name=file, path_output=outpath3)

    stop=time.perf_counter()
    print(f"File {file} processed in: {stop-start}")

    temp_dir.cleanup()

#Call to ECGScan

list_of_files = os.listdir(outpath3)
files=[x for x in list_of_files if x.endswith(".png") or x.endswith(".jpg")]

print (f"\nNow open preprocessed Images with ECGScan")

ECGScan = win32com.client.Dispatch("ECGScan.Document")
ret = ECGScan.ShowWin
for count,file in enumerate(files):
    print(f"\n\tECGScan is processing file: {file}, \n\t{count+1}/{len(files)}")
    path_to_img = str(outpath3 + str(file))
    ECGScan.FileName = path_to_img

    go_on=input("\tPress any button to go on... ")

    ECGScan.SaveFDA(str(outpath4 + file.split(sep=".")[0] + "xml"))

ECGScan.CloseApplication




