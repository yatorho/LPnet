import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np

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
ap.add_argument("-w", "--write_file", default='res101.out',
                help="file for output")

args = vars(ap.parse_args())

use_gpu = torch.cuda.is_available()
print("Use GPU: ", use_gpu)

num_classes = 4
img_size = (480, 480)
batch_size = int(args["batchsize"]) if use_gpu else 8

model_folder = 'Res101/'
store_name = model_folder + 'res101.pth'

if not os.path.isdir(model_folder):
    os.mkdir(model_folder)

epochs = int(args["epochs"])

with open(args['write_file'], 'wb') as out_f:
    pass

print("===============================================")
print("Starting training with:\nInput folder: {} \n Epochs: {} \n Batch size: {} \n Image size: {} \n Classes: {} \n Model Store: {} \n Resume: {} \n Write File: {}".format(
    args["images"], epochs, batch_size, img_size, num_classes, store_name, args['resume'], args['write_file']))
print("===============================================")

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

model = Res101(num_classes=num_classes)
if use_gpu:
    model = model.cuda()

resume_file = str(args["resume"])
if args['resume'] != '0':
    if not os.path.isfile(args['resume']):
        print("Error: no checkpoint directory found!")
        exit()
    print("Loading checkpoint: {}".format(args['resume']))
    checkpoint = torch.load(args['resume'])
    model.load_state_dict(torch.load(resume_file))
else:
    print("No checkpoint found. Starting training from scratch.")

# print(model)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable parameters: {}".format(n_params))

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

train_dst = data_loader.ChaLocDataLoader(args["images"].split(","), img_size)
val_dst = data_loader.ChaLocDataLoader(args["val_images"].split(","), img_size)
train_loader = torch.utils.data.DataLoader(
    train_dst, batch_size=batch_size, shuffle=True)
val_batch_size = 16
val_loader = torch.utils.data.DataLoader(
    val_dst, batch_size=val_batch_size, shuffle=False)

def validate(model, epoch):
    model.eval()
    loss_avg = []
    start = time.time()
    for i, (images, labels) in enumerate(val_loader):
        if use_gpu:
            x = torch.autograd.Variable(images.cuda(), requires_grad=False)
            y = torch.autograd.Variable(labels.cuda(), requires_grad=False)
        else:
            x = torch.autograd.Variable(images, requires_grad=False)
            y = torch.autograd.Variable(labels, requires_grad=False)
        
        y_pred = model(x)
        
        if len(y_pred) == val_batch_size:
            loss = criterion(y_pred, y)
            assert loss.shape == torch.Size([])

            loss_avg.append(loss.item())
        
        if i % 50 == 1:
            with open(args['write_file'], 'a') as out_f:
                out_f.write("Validation: Batch: {}/{} \t Loss: {} \t Time: {} \t Num images: {}\t \n".format(i, len(val_loader), np.mean(loss_avg[-50:]), time.time() - start, i*val_batch_size))
    print("Validation: Epoch: {}/{} \t Loss: {} \t Time: {} \t Num images: {}\t ".format(epoch+1, epochs, np.mean(loss_avg), time.time() - start, len(val_loader)*val_batch_size))


def train(model, loss, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        loss_avg = []
        start = time.time()

        print("Epoch: {}/{} with lr: {}...".format(epoch+1, epochs, scheduler.get_last_lr()))
        for i, (images, labels) in enumerate(train_loader):
            if use_gpu:
                x = torch.autograd.Variable(images.cuda(), requires_grad=False)
                y = torch.autograd.Variable(labels.cuda(), requires_grad=False)
            else:
                x = torch.autograd.Variable(images, requires_grad=False)
                y = torch.autograd.Variable(labels, requires_grad=False)
            
            y_pred = model(x)
            
            if len(y_pred) == batch_size:
                loss = criterion(y_pred, y)
                assert loss.shape == torch.Size([])

                loss_avg.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if i % 50 == 1:
                # torch.save(model.state_dict(), store_name)
                with open(args['write_file'], 'a') as out_f:
                    out_f.write("Epoch: {}/{} \t Batch: {}/{} \t Loss: {} \t Time: {} \t Num images: {}\t \n".format(
                        epoch+1, epochs, i, len(train_loader), np.mean(loss_avg[-50:]), time.time()-start, i * batch_size))
        print("Epoch: {}/{} \t Loss: {}".format(epoch+1, epochs, np.mean(loss_avg)))
        with open(args['write_file'], 'a') as out_f:
            out_f.write("Epoch: {}/{} \t Loss: {} \t Time: {} \t\n".format(
                epoch+1, epochs, np.mean(loss_avg), time.time()-start))
        torch.save(model.state_dict(), store_name + str(epoch+1))
        scheduler.step()

        validate(model, epoch)
    return model

if __name__ == "__main__":
    train(model, criterion, optimizer, epochs)