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
img_size = (480, 480)
batch_size = int(args["batchsize"]) if use_gpu else 8

model_folder = 'Res50/'
store_name = model_folder + 'res50.pth'

if not os.path.isdir(model_folder):
    os.mkdir(model_folder)

epochs = int(args["epochs"])

with open(args['write_file'], 'wb') as out_f:
    pass

print("===============================================")
print("Starting training with:\nInput folder: {} \n Epochs: {} \n Batch size: {} \n Image size: {} \n Classes: {} \n Model Store: {} \n Resume: {} \n Write File: {}".format(
    args["images"], epochs, batch_size, img_size, num_classes, store_name, args['resume'], args['write_file']))
print("===============================================")

class Res50(nn.Module):
    def __init__(self, num_classes=4):
        super(Res50, self).__init__()
        self.res50 = models.resnet50(pretrained=True).float()
        # Set res50 dtype to float32
        self.res50.fc = nn.Linear(2048, num_classes).float()

    def forward(self, x):
        out = self.res50(x)
        assert out.size(1) == num_classes
        return out

model = Res50(num_classes=num_classes)
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

print(model)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable parameters: {}".format(n_params))

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

dst = data_loader.ChaLocDataLoader(args["images"].split(","), img_size)
train_loader = torch.utils.data.DataLoader(
    dst, batch_size=batch_size, shuffle=True)

def train(model, loss, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        loss_avg = []
        start = time.time()

        print("Epoch: {}/{} ...".format(epoch+1, epochs))
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
    
    return model

if __name__ == "__main__":
    train(model, criterion, optimizer, epochs)