import glob
import os
import cv2
from tqdm import tqdm

PATH_IMAGES = "./dataset/images"
RESIZED_SIZE = 256

# please check folder named "resized" doesn't exist is the current directory.


# load images path as list
def load_images_path_as_list():
    folder_list = glob.glob(PATH_IMAGES+"/*")
    files_list = glob.glob(PATH_IMAGES+"/**/*")

    return folder_list, files_list


# make save directory and, resize and save images.
def resize_and_save(folder_path, image_path):
    if not os.path.exists("./resized"):
        os.mkdir("./resized")
        os.mkdir("./resized/images")
        for folder in folder_path:
            folder = folder.replace("./dataset", "")
            os.mkdir("./resized"+folder)
            print("made directory:"+"./resized"+folder)

    print("Directories made.")
    print("resizing images...")
    for image in tqdm(image_path):
        img = cv2.imread(image)
        img = cv2.resize(img, (RESIZED_SIZE, RESIZED_SIZE))
        save_path = image.replace("./dataset", "./resized")
        cv2.imwrite(save_path, img)

    print("All images saved.")
if __name__=='__main__':
    folder_list, image_list = load_images_path_as_list()
    resize_and_save(folder_list, image_list)