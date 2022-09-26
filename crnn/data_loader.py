import cv2
import torch
from imutils import paths
import numpy as np


class ChaLocDataLoader(torch.utils.data.Dataset):
    def __init__(self, img_dir, img_size) -> None:
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]

        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]

        ori_img = cv2.imread(img_name)
        img = cv2.resize(ori_img, self.img_size)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)
        img /= 255.0

        # Use '\\' to split the path in Windows, '/' in Linux
        lbl = img_name.rsplit('\\', 1)[-1].rsplit('.', 1)[0].split('-')
        [left_up, right_down] = [
            [int(eel) for eel in e.split('&')] for e in lbl[2].split('_')]

        ori_w, ori_h = float(ori_img.shape[1]), float(ori_img.shape[0])

        assert ori_img.shape == (1160, 720, 3)
        lbl = [(left_up[0] + right_down[0]) / (2 * ori_w), (left_up[1] + right_down[1]) /
               (2 * ori_h), (right_down[0] - left_up[0]) / ori_w, (right_down[1] - left_up[1]) / ori_h]

        return img.astype(np.float32), np.array(lbl).astype(np.float32)

class ChaLPFromOriDataLoader(torch.utils.data.Dataset):
    def __init__(self, img_dir, img_size):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_name = self.img_paths[index]

        ori_img = cv2.imread(img_name)
        img = cv2.resize(ori_img, self.img_size)

        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)
        img /= 255.0

        # Use '\\' to split the path in Windows, '/' in Linux
        lbl = img_name.rsplit('\\', 1)[-1].rsplit('.', 1)[0].split('_')
        lbl = [int(el) for el in lbl]
        assert len(lbl) == 7

        return img.astype(np.float32), np.array(lbl).astype(np.float32)
