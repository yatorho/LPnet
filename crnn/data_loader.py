import cv2
import torch
from imutils import paths
import numpy as np


ctc_provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学"]
ctc_ads = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ctc_table = ctc_provinces + ctc_ads

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

        return img.astype(np.float32), np.array(lbl).astype(np.int32)

class ChaLpFromPicDataLoader(torch.utils.data.Dataset):
    def __init__(self, img_dir, img_size: tuple=(3, 480, 480)):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        self.img_size = img_size
    
    def __len__(self) -> int:
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_name = self.img_paths[index]
        name = img_name.rsplit('\\', 1)[-1].rsplit('.', 1)[0]
        assert len(name) == 7, name
        lbl = [ctc_table.index(el) for el in name]

        ori_img = cv2.imdecode(np.fromfile(img_name, dtype=np.uint8), -1)
        img = cv2.resize(ori_img, self.img_size)
        assert img.shape == (480, 480, 3)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)
        img /= 255.0

        return img.astype(np.float32), np.array(lbl).astype(np.int32)

