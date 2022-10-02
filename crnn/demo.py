import argparse
import time
import os

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import data_loader

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--img_dir", required=True, help="Path to the image directory")
ap.add_argument("-w1", "--weights1", required=True, help="Path to the ResNet101 weights file")
ap.add_argument("-w2", "--weights2", required=True, help="Path to the CRNN weights file")
ap.add_argument("-s", "--save_dir", default="0", help="Path to the save directory")
ap.add_argument("-c", "--cuda", default="0", help="Use cuda or not")
ap.add_argument("-b", "--batch_size", default="1", help="Batch size")
ap.add_argument("-sp", "--show_picture", default="0", help="Show picture or not")
ap.add_argument("-r", "--roi_expand_ratio", default=0.01, help="The ratio of the roi size to the original size")

args = vars(ap.parse_args())

use_gpu = bool(int(args["cuda"])) and torch.cuda.is_available()
batch_size = int(args["batch_size"])
show_picture = bool(int(args["show_picture"]))
save = True if args["save_dir"] != "0" else False
save_dir = None
if save:
    save_dir = args["save_dir"]
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

img_size = (480, 480)
num_classes = 4

# roi_w = roi_w * (1 + 2 * roi_expand_ratio)
# roi_h = roi_h * (1 + 2 * roi_expand_ratio)
# This makes the roi size to be larger than the original size
roi_expand_ratio = float(args["roi_expand_ratio"])

ctc_provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学"]
ctc_ads = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ctc_table = ctc_provinces + ctc_ads
ctc_table.append('-')  # use '-' as blank

class Res101(nn.Module):
    def __init__(self, num_classes=4):
        super(Res101, self).__init__()
        self.res101 = models.resnet101(pretrained=False).float()
        # Set res101 dtype to float32
        self.res101.fc = nn.Linear(2048, num_classes).float()

    def forward(self, x):
        out = self.res101(x)
        assert out.size(1) == num_classes
        return torch.sigmoid(out)

class CRnn(nn.Module):
    def __init__(self):
        super(CRnn, self).__init__()
        # Build a CRNN model here
        # input size is (bs, 3, 24, 94), build a CNN model whose output size is (bs, 512, 18, 1)
        self.cnn = cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), # (bs, 64, 24, 94)
            nn.Conv2d(32, 64, 3, 1, 1), # (bs, 64, 24, 94)
            nn.MaxPool2d(2, 2), # (bs, 64, 12, 47)
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 1, 1), # (bs, 128, 12, 47)
            nn.Conv2d(128, 256, 3, 2, 1), # (bs, 256, 6, 24)
            nn.AvgPool2d(2, 2), # (bs, 256, 3, 12),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 3, (1, 2), 1), # (bs, 512, 3, 6)
            nn.Conv2d(512, 512, 3, (1, 1), 1), # (bs, 512, 3, 6)
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )  # The output size should be (bs, 512, 3, 6)
        self.rnn1 = nn.LSTM(512, 256, 2, bidirectional=True)
        self.rnn2 = nn.LSTM(512, 256, 2, bidirectional=True)
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, len(ctc_table))

        
    def forward(self, x): # (b, 3, 94, 24)
        x = self.cnn(x).reshape(x.shape[0], 512, -1) # (b, 512, 18)
        x = x.permute(2, 0, 1) # (18, b, 512)
        x, _ = self.rnn1(x) # (18, b, 512)
        x, _ = self.rnn2(x) # (18, b, 512)
        x = self.fc1(x) # (18, b, 1024)
        x = self.fc2(x) # (18, b, len(ctc_table)
        return x.log_softmax(2)

# Load model
res101 = Res101()
crnn = CRnn()
res101.load_state_dict(torch.load(args["weights1"]))
crnn.load_state_dict(torch.load(args["weights2"]))
if use_gpu:
    res101 = res101.cuda()
    crnn = crnn.cuda()

res101.eval()
crnn.eval()

# Load data
dst = data_loader.ChaLpFromPicDataLoader(args["img_dir"].split(","), img_size)
data_loader = data.DataLoader(dst, batch_size=batch_size, shuffle=False, num_workers=0)

def roi_from_res101(imgs, labels):
    # Get the roi from res101's output
    # imgs: (bs, 3, 480, 480)
    # labels: (bs, 4)
    # return: (bs, 3, 24, 94)
    imgs = imgs.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    rois = []
    for i in range(imgs.shape[0]):
        label = labels[i]
        img = imgs[i]
        x = label[0] * img_size[0]
        y = label[1] * img_size[1]
        w = label[2] * img_size[0]
        h = label[3] * img_size[1]
        split_x_from = x - (1 + roi_expand_ratio) * w / 2 if x - (1 + roi_expand_ratio) * w / 2 > 0 else 0
        split_x_to = x + (1 + roi_expand_ratio) * w / 2 if x + (1 + roi_expand_ratio) * w / 2 < img_size[0] else img_size[0]
        split_y_from = y - (1 + roi_expand_ratio) * h / 2 if y - (1 + roi_expand_ratio) * h / 2 > 0 else 0
        split_y_to = y + (1 + roi_expand_ratio) * h / 2 if y + (1 + roi_expand_ratio) * h / 2 < img_size[1] else img_size[1]
        roi = img[:, int(split_y_from):int(split_y_to), int(split_x_from):int(split_x_to)].transpose(1, 2, 0)
        roi = cv2.resize(roi, (94, 24))
        roi = roi.transpose(2, 0, 1)
        rois.append(roi)
    res = torch.from_numpy(np.array(rois)).float().cuda() if use_gpu else torch.from_numpy(np.array(rois)).float()
    return res

def ctc_decode(preds): # (18, bs, len(ctc_table))
    out2 = preds.permute(1, 0, 2).argmax(2) # (bs, 18)
    
    preds = out2.cpu().numpy()
    preb_labels = [] # (bs, T)

    for j in range(preds.shape[0]):
        pred = preds[j]
        not_repeat_blank_label = []

        pre_c = pred[0]
        if pre_c != len(ctc_table) - 1:
            not_repeat_blank_label.append(pre_c)
        for c in pred[1:]:
            if (pre_c == c) or (c == len(ctc_table) - 1):
                if c == len(ctc_table) - 1:
                    pre_c = c
                continue
            not_repeat_blank_label.append(c)
            pre_c = c
        preb_labels.append(not_repeat_blank_label)
    return preb_labels # (bs, T)

def ctc_char_decode(preds): # (18, bs, len(ctc_table))
    preb_labels = ctc_decode(preds) # (bs, T)
    preb_labels = [[ctc_table[c] for c in label] for label in preb_labels]
    return preb_labels
    
def validate():
    total_nums = 0
    correct_nums = 0
    start_time = time.time()
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(data_loader):
            if use_gpu:
                imgs = imgs.cuda()
                labels = labels.cuda()
            out1 = res101(imgs)
            out2 = crnn(roi_from_res101(imgs, out1)) # (18, bs, len(ctc_table))

            preb_labels = ctc_decode(out2) # (bs, T)
            actual_labels = labels = labels.cpu().numpy().tolist() # (bs, T)
            # assert len(preb_labels) == len(actual_labels) and imgs.shape[0] == batch_size

            batch_correct_num = 0
            for j in range(imgs.shape[0]):
                if preb_labels[j] == actual_labels[j]:
                    batch_correct_num += 1
            total_nums += imgs.shape[0]
            correct_nums += batch_correct_num

            chars = ctc_char_decode(out2)
            if save or show_picture:
                for j in range(imgs.shape[0]):
                    cx = out1[j][0].item() * img_size[0]
                    cy = out1[j][1].item() * img_size[1]
                    cw = out1[j][2].item() * img_size[0]
                    ch = out1[j][3].item() * img_size[1]

                    img = imgs[j].cpu().detach().numpy().transpose(1, 2, 0)
                    # Enable chinese text for cv2.putText
                    pil_img = Image.fromarray(cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_img)
                    font = ImageFont.truetype("simhei.ttf", 30, encoding="utf-8")
                    draw.text((cx - cw / 2, cy + ch / 2), "".join(chars[j]), (255, 0, 0), font=font)
                    img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
                    # ROI rectangle
                    img = cv2.rectangle(img, (int(cx - cw / 2), int(cy - ch / 2)), (int(cx + cw / 2), int(cy + ch / 2)), (0, 255, 0), 2)

                    if show_picture:
                        cv2.imshow("img", img)
                        cv2.waitKey(0)

                    if save:
                        # file name is the actual label. We should support chinese file name.
                        cv2.imencode(".jpg", img)[1].tofile(os.path.join(args["save_dir"], "".join([ctc_table[c] for c in actual_labels[j]]) + ".jpg"))


            print("Batch: %d \t Correct: %d \t Size: %d \t Acc: %5.3f%%" % (i, correct_nums, imgs.shape[0], batch_correct_num / imgs.shape[0] * 100))
            for j in range(imgs.shape[0]):
                print('P{}: \t Actual:{} \t Preb: {}'.format(j, "".join([ctc_table[c] for c in actual_labels[j]]), "".join(chars[j])))
        print("============================================")
        print("Total nums: %d \t Correct nums: %d \t Acc: %5.3f%%" % (total_nums, correct_nums, correct_nums / total_nums * 100))
        print("All elapsed time: %.3fs \t Avg time: %.3fs" % (time.time() - start_time, (time.time() - start_time) / total_nums))

if __name__ == '__main__':
    validate()
