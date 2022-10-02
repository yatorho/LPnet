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
ap.add_argument("-v", "--val_images", required=True,
                help="path to the val file")
ap.add_argument("-n", "--epochs", default=25,
                help="epochs for trains")
ap.add_argument("-b", "--batchsize", default=4,
                help="batch size for train")
ap.add_argument("-r", "--resume", default='0',
                help="file for re-train")
ap.add_argument("-w", "--write_file", default='crnn.out',
                help="file for output")
args = vars(ap.parse_args())

use_gpu = torch.cuda.is_available()
print("Use GPU: ", use_gpu)

num_classes = 4
img_size = (94, 24) 
input_size = (img_size[1], img_size[0])
batch_size = int(args["batchsize"]) if use_gpu else 8

model_folder = 'CRnnv2/'
store_name = model_folder + 'crnn.pth'

if not os.path.isdir(model_folder):
    os.mkdir(model_folder)

epochs = int(args["epochs"])

with open(args['write_file'], 'wb') as out_f:
    pass

print("===============================================")
print("Starting training with:\nInput folder: {} \n Epochs: {} \n Batch size: {} \n Image size: {} \n Classes: {} \n Model Store: {} \n Resume: {} \n Write File: {}".format(
    args["images"], epochs, batch_size, img_size, num_classes, store_name, args['resume'], args['write_file']))
print("===============================================")

ctc_provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学"]
ctc_ads = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ctc_table = ctc_provinces + ctc_ads
ctc_table.append('-')  # use '-' as blank

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

resume_file = str(args['resume'])
if resume_file != '0':
    if not os.path.isfile(resume_file):
        print("=> no checkpoint found at '{}'".format(resume_file))
        exit(0)
    print("=> loading checkpoint '{}'".format(resume_file))

    model = CRnn()
    model.load_state_dict(torch.load(resume_file))
    if use_gpu:
        model = model.cuda()
else:
    model = CRnn()
    if use_gpu:
        model = model.cuda()

print(model)
n_params = sum([p.numel() for p in model.parameters()])
print('Number of parameters: {}'.format(n_params))

ctc_loss = nn.CTCLoss(blank=len(ctc_table) - 1, reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=1e-3)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.98)

train_dst = data_loader.ChaLPFromOriDataLoader(args["images"].split(","), img_size)
train_loader = torch.utils.data.DataLoader(
    train_dst, batch_size=batch_size, shuffle=True)

val_batch_size = 16

val_dst = data_loader.ChaLPFromOriDataLoader(args["val_images"].split(","), img_size)
val_loader = torch.utils.data.DataLoader(
    val_dst, batch_size=val_batch_size, shuffle=True)

def validate(model, epoch):
    model.eval()

    loss_avg = []
    start = time.time()

    total_nums = 0
    correct_nums = 0

    for i, (imgs, labels) in enumerate(val_loader):
        if use_gpu:
            x = torch.autograd.Variable(imgs.cuda(), requires_grad=False)
            y = torch.autograd.Variable(labels.cuda(), requires_grad=False).long()
        else:
            x = torch.autograd.Variable(imgs, requires_grad=False)
            y = torch.autograd.Variable(labels, requires_grad=False).long()
        
        y_pred = model(x)  # (18, b, len(ctc_table)
        loss = ctc_loss(y_pred, y, torch.full((y.shape[0],), y_pred.shape[0], dtype=torch.long), torch.full((y.shape[0],), y.shape[1], dtype=torch.long))
        loss_avg.append(loss.item())
 
        preds = y_pred.permute(1, 0, 2).argmax(2)  # (b, 18)
        preds = preds.cpu().numpy()
        preb_labels = [] # (b, T)

        for i in range(preds.shape[0]):
            preb = preds[i]
            no_repeat_blank_label = []

            pre_c = preb[0]
            if pre_c != len(ctc_table) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb[1:]:
                if (pre_c == c) or (c == len(ctc_table) -1):
                    if c == len(ctc_table) -1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
        actual_labels = labels.cpu().numpy().tolist()

        assert len(preb_labels) == len(actual_labels)
        
        for i in range(len(preb_labels)):
            if preb_labels[i] == actual_labels[i]:
                correct_nums += 1
            total_nums += 1
        
        if i % 50 == 1:
            with open(args['write_file'], 'a') as out_f:
                out_f.write("Validation: Batch: {}/{} \t Acc: {} \t Loss: {} \t Time: {} \t Num images: {}\t \n".format(i, len(val_loader), 
                    correct_nums/total_nums, np.mean(loss_avg), time.time() - start, len(val_loader)*val_batch_size))

    with open(args['write_file'], 'a') as out_f:
        out_f.write("Validation: Epoch: {}/{} \t Acc: {} \t Loss: {} \t Time: {} \t Num images: {}\t \n".format(epoch+1, epochs, 
            correct_nums/total_nums, np.mean(loss_avg), time.time() - start, len(val_loader)*val_batch_size))
    print("Validation: Epoch: {}/{} \t Acc: {} \t Loss: {} \t Time: {} \t Num images: {}\t ".format(epoch+1, epochs, 
        correct_nums/total_nums, np.mean(loss_avg), time.time() - start, len(val_loader)*val_batch_size))

def train(model):
    for epoch in range(epochs):
        model.train()
        loss_avg = []
        start = time.time()

        print("Epoch: {}/{} with lr: {}...".format(epoch + 1, epochs, lr_scheduler.get_last_lr()))
        for i, (imgs, labels) in enumerate(train_loader):
            if use_gpu:
                x = torch.autograd.Variable(imgs.cuda(), requires_grad=False)
                y = torch.autograd.Variable(labels.cuda(), requires_grad=False).long()
            else:
                x = torch.autograd.Variable(imgs, requires_grad=False)
                y = torch.autograd.Variable(labels, requires_grad=False).long()
            
            y_pred = model(x)  # (18, b, len(ctc_table)

            if y_pred.shape[1] == batch_size:
                # 7 is the max length of the label. For different dataset, you need to change this number
                assert y.shape == (batch_size, 7) 
                
                loss = ctc_loss(y_pred, y, torch.full((batch_size,), y_pred.shape[0], dtype=torch.long), torch.full((batch_size,), y.shape[1], dtype=torch.long))
                loss_avg.append(loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if i % 50 == 1:
                # torch.save(model.state_dict(), store_name)
                with open(args['write_file'], 'a') as out_f:
                    out_f.write("Epoch: {}/{} \t Batch: {}/{} \t Loss: {} \t Time: {} \t Num images: {}\t \n".format(
                        epoch+1, epochs, i, len(train_loader), np.mean(loss_avg[-50:]), time.time()-start, i * batch_size))
                pass

        print("Epoch: {}/{} \t Loss: {} \t ".format(epoch+1, epochs, np.mean(loss_avg)))
        with open(args['write_file'], 'a') as out_f:
            out_f.write("Epoch: {}/{} \t Loss: {} \t Time: {} \t\n".format(
                epoch+1, epochs, np.mean(loss_avg), time.time()-start))
        if epoch % 10 == 0:
            torch.save(model.state_dict(), store_name + str(epoch+1))
        lr_scheduler.step()

        validate(model, epoch)
    

if __name__ == '__main__':
    train(model)
    # validate(model, 0)

