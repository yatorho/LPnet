import torch
import cv2
import numpy as np
import torch.nn as nn
import torchvision.models as models


class Loc(nn.Module):
    def __init__(self, num_classes=4):
        super(Loc, self).__init__()
        # input (3, 480, 480)
        hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48,
                      kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64,
                      kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160,
                      kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192,
                      kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192,
                      kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192,
                      kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192,
                      kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden9 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden10 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self.features = nn.Sequential(
            hidden1,
            hidden2,
            hidden3,
            hidden4,
            hidden5,
            hidden6,
            hidden7,
            hidden8,
            hidden9,
            hidden10
        )  # (?, 192, 11, 11)
        self.classifier = nn.Sequential(
            nn.Linear(192 * 11 * 11, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        out = self.features(x)  # (?, 192, 11, 11)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out

class Res50(nn.Module):
    def __init__(self, num_classes=4):
        super(Res50, self).__init__()
        self.res50 = models.resnet50(pretrained=True).float()
        # Set res50 dtype to float32
        self.res50.fc = nn.Linear(2048, num_classes).float()

    def forward(self, x):
        out = self.res50(x)
        return out

# model = Loc(4).cuda()
model = Res50(4).cuda()

model.load_state_dict(torch.load('Res50/res50.pth1'))


ori_img = cv2.imread('ps\\3.jpg')
ori_img = cv2.resize(ori_img, (480, 480))
img = np.array(ori_img)
img = img.transpose(2, 0, 1)
assert img.shape == (3, 480, 480)
assert img.dtype == np.uint8
input = torch.tensor(img.reshape(1, 3, 480, 480).astype('float32') / 255.).cuda()

out = model(input)
assert out.data.cpu().numpy().shape == (1, 4)

[cx, cy, w, h] = out.data.cpu().numpy()[0].tolist()

left_up = [(cx - w/2)*img.shape[2], (cy - h/2)*img.shape[1]]
right_down = [(cx + w/2)*img.shape[2], (cy + h/2)*img.shape[1]]

cv2.rectangle(ori_img, (int(left_up[0]), int(left_up[1])), (int(right_down[0]), int(right_down[1])), (0, 0, 255), 2)
cv2.imshow('ori_img', ori_img)
cv2.waitKey(0)
