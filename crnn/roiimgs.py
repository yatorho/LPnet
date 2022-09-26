# Crop the images of CCPD dataset to an Area of Interest (AOI) and save them to a new directory.
# Usage: python3 roiimgs.py <source directory> <destination directory> -n <number of files> -h <height of AOI> -w <width of AOI>

import os
import sys
import shutil
import random
import cv2


def crop_images(source_dir, dest_dir, num_files, size):
    files = os.listdir(source_dir)
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    random.shuffle(files)
    for i in range(num_files):
        img = cv2.imread(os.path.join(source_dir, files[i]))
        
        # Use '\\' to split the path in Windows, '/' in Linux
        lbl = files[i].rsplit('\\', 1)[-1].rsplit('.', 1)[0].split('-')
        [left_up, right_down] = [
            [int(eel) for eel in e.split('&')] for e in lbl[2].split('_')]
        
        img = img[left_up[1]:right_down[1], left_up[0]:right_down[0], :]
        img = cv2.resize(img, size)
        assert img.shape == (size[1], size[0], 3)

        file_name = files[i].rsplit('\\', 1)[-1].rsplit('.', 1)[0].rsplit('-')[4] + '.jpg'
        cv2.imwrite(os.path.join(dest_dir, file_name), img)


if __name__ == "__main__":
    if len(sys.argv) != 9:
        print("Usage: python3 roiimgs.py <source directory> <destination directory> -n <number of files> -h <height of AOI> -w <width of AOI>")
        sys.exit(1)
    sd = sys.argv[1]
    dd = sys.argv[2]
    n = int(sys.argv[4])
    size = (int(sys.argv[8]), int(sys.argv[6]))
    crop_images(sd, dd, n, size)
