import os
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# def resize_backgrounds(path, width=1920, height=1080):
#     for image in os.listdir(path):
#         full_path = os.path.join(path, image)
#         with Image.open(full_path) as img:
#             resized = img.resize((width, height))
#             resized.save(full_path)

# def return_mask(image):
#     colours = np.array([255, 0, 255])
#     mask = Image.fromarray(cv2.inRange(np.array(image), colours, colours))
#     return mask

def process_folder(folder, chessboards_path, save_path):
    folder_path = os.path.join(chessboards_path, folder)
    save_folder = os.path.join(save_path, folder)
    os.makedirs(save_folder, exist_ok=True)
    colours = np.array([255, 0, 255])
    for img in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img)
        image = Image.open(img_path)
        mask = Image.fromarray(cv2.inRange(np.array(image), colours, colours))
        mask_path = os.path.join(save_folder, f"{os.path.splitext(img)[0]}.png")
        mask.save(mask_path)

def main():
    base_directory = os.path.dirname(os.path.dirname(os.getcwd()))
    background_path = os.path.join(base_directory, 'backgrounds')
    chessboards_path = os.path.join(base_directory, 'Rendered_Data', 'Auto_generated', 'training','images')
    save_path = os.path.join(os.path.dirname(chessboards_path), 'images_masks')

    folders = os.listdir(chessboards_path)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(process_folder, folders, [chessboards_path]*len(folders), [save_path]*len(folders)), desc="Processing", total=len(folders)))

if __name__ == "__main__":
    main()