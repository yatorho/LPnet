from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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
ap.add_argument("-w", "--write_file", default='loc.out',
                help="file for output")

args = vars(ap.parse_args())

use_gpu = torch.cuda.is_available()
print("Use GPU: ", use_gpu)

num_classes = 4
img_size = (480, 480)
batch_size = int(args["batchsize"]) if use_gpu else 8

model_folder = 'Loc/'
store_name = model_folder + 'loc.pth'
if not os.path.isdir(model_folder):
    os.mkdir(model_folder)

epochs = int(args["epochs"])

with open(args['write_file'], 'wb') as out_f:
    pass

print("===============================================")
print("Starting training with:\nInput folder: {} \n Epochs: {} \n Batch size: {} \n Image size: {} \n Classes: {} \n Model Store: {} \n Resume: {} \n Write File: {}".format(
    args["images"], epochs, batch_size, img_size, num_classes, store_name, args['resume'], args['write_file']))
print("===============================================")

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
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out #?,4


epoch = 0
resume_file = str(args["resume"])
if resume_file != '0':
    if not os.path.isfile(resume_file):
        print("=> no checkpoint found at '{}'".format(resume_file))
        exit(0)
    print("=> loading checkpoint '{}'".format(resume_file))

    loc_model = Loc(num_classes)
    loc_model.load_state_dict(torch.load(resume_file))
    if use_gpu:
        loc_model = loc_model.cuda()
else:
    loc_model = Loc(num_classes)
    if use_gpu:
        loc_model = loc_model.cuda()

print(loc_model)
n_params = sum([p.numel() for p in loc_model.parameters()])
print('Number of parameters: {}'.format(n_params))

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(loc_model.parameters(), lr=0.001)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.6)


dst = data_loader.ChaLocDataLoader(args["images"].split(","), img_size)
train_loader = torch.utils.data.DataLoader(
    dst, batch_size=batch_size, shuffle=True)


def train_model(model, loss, optimizer, num_epochs=epochs):
    for epoch in range(num_epochs):
        loss_avg = []
        print('Epoch {}/{}.... See loc.out for more info'.format(epoch, num_epochs - 1))
        model.train(True)
        start = time.time()

        for i, (inputs, labels) in enumerate(train_loader):
            # labels = np.array([el.numpy() for el in labels])
            # assert(inputs.shape == torch.Size([batch_size, 3, img_size[0], img_size[1]]))
            # assert(labels.shape == torch.Size([batch_size, 4]))

            if use_gpu:
                x = torch.autograd.Variable(inputs.cuda(), requires_grad=False)
                y = torch.autograd.Variable(labels.cuda(), requires_grad=False)
            else:
                x = torch.autograd.Variable(inputs, requires_grad=False)
                y = torch.autograd.Variable(labels, requires_grad=False)

            y_pred = model(x)

            if len(y_pred) == batch_size:
                loss = 0.8 * nn.L1Loss().cuda()(y_pred[:, :2], y[:, :2]) + 0.2 * nn.L1Loss(
                ).cuda()(y_pred[:, 2:], y[:, 2:])

                assert loss.shape == torch.Size([])
                loss_avg.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                torch.save(model.state_dict(), store_name)

            if i % 50 == 1:
                with open(args["write_file"], 'a') as f:
                    f.write("train %s images, use %s seconds, loss %s\n" % (
                        i * batch_size, time.time() - start, np.mean(loss_avg[-50:])))
        lr_scheduler.step()

        print("%s %s %s\n" % (epoch, np.mean(loss_avg), time.time() - start))
        with open(args["write_file"], 'a') as f:
            f.write("%s %s %s\n" % (epoch, np.mean(loss_avg), time.time() - start))
        torch.save(model.state_dict(), store_name + str(epoch))
    return model

if __name__ == '__main__':
    train_model(loc_model, loss, optimizer, num_epochs=epochs)
