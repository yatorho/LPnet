# Random rename file in CCPD dataset to chinese file name.
# Useage: python ccpd_char_extra.py <CCPD dataset path> <save path> -n <number of files>

import os
import sys
import random
import shutil


from imutils import paths



ccpd_provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ccpd_alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ccpd_ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

def rename_file(dataset_path, save_path, num_files):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_path = [p for p in paths.list_images(dataset_path)]
    random.shuffle(image_path)

    for i in range(num_files):
        img_name = image_path[i]
        lbl = img_name.rsplit('\\', 1)[-1].rsplit('.', 1)[0].split('-')[-3]
        lbl = lbl.split('_')

        save_file = [ccpd_provinces[int(lbl[0])]]
        for i in range(1, 7):
            save_file.append(ccpd_ads[int(lbl[i])])
        save_file = ''.join(save_file) + '.jpg'
        save_file = os.path.join(save_path, save_file)
        # copy file
        shutil.copy(img_name, save_file)

if __name__ == '__main__':
    dataset_path = sys.argv[1]
    save_path = sys.argv[2]
    num_files = int(sys.argv[4])
    rename_file(dataset_path, save_path, num_files)
