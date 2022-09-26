import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
from imutils import paths

import os
import argparse
import time

import data_loader

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
                help="path to the input file")
ap.add_argument("-n", "--epochs", default=25,
                help="epochs for trains")
ap.add_argument("-b", "--batchsize", default=4,
                help="batch size for train")
ap.add_argument("-r", "--resume", default='0',
                help="file for re-train")
ap.add_argument("-w", "--write_file", default='res50.out',
                help="file for output")
args = vars(ap.parse_args())

use_gpu = torch.cuda.is_available()
print("Use GPU: ", use_gpu)

num_classes = 4
img_size = (500, 200)  # 500 for width, 200 for height
input_size = (img_size[1], img_size[0])
batch_size = int(args["batchsize"]) if use_gpu else 8

model_folder = 'CTC/'
store_name = model_folder + 'ctc.pth'

if not os.path.isdir(model_folder):
    os.mkdir(model_folder)

epochs = int(args["epochs"])

with open(args['write_file'], 'wb') as out_f:
    pass

print("===============================================")
print("Starting training with:\nInput folder: {} \n Epochs: {} \n Batch size: {} \n Image size: {} \n Classes: {} \n Model Store: {} \n Resume: {} \n Write File: {}".format(
    args["images"], epochs, batch_size, img_size, num_classes, store_name, args['resume'], args['write_file']))
print("===============================================")

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学"]
ads = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
       'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'I', 'O']
ctc_table = provinces + ads
ctc_table.append('-')  # use '-' as blank


class CTC(nn.Module):
    def __init__(self):
        super(CTC, self).__init__()
        # Build a CRNN model here
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.rnn = nn.Sequential(
            nn.LSTM(512, 256, bidirectional=True),
            nn.LSTM(512, 256, bidirectional=True)
        )
        # CTC layer
        self.fc = nn.Linear(512, len(ctc_table) + 1)

        
    def forward(self, x):
        conv = self.conv(x)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        r = self.rnn(conv.squeeze(2))
        r = r.permute(1, 0, 2)
        r = r.contiguous().view(-1, r.size(2))
        r = self.fc(r)
        return r

if __name__ == '__main__':
    dst = data_loader.ChaLPFromOriDataLoader(args["images"].split(","), img_size)
    print(dst.img_paths[0])
    print(dst[0][0].shape)
    print(dst[0][1].shape)