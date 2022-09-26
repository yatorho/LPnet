import torch

import os
import argparse

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

# The shape of ROI is (?, 3, 200, 500)
img_size = (200, 500)
batch_size = int(args["batchsize"]) if use_gpu else 8
epochs = int(args["epochs"])

model_folder = 'ChaFROI/'
store_name = model_folder + 'chafroi.pth'
if not os.path.isdir(model_folder):
    os.mkdir(model_folder)


with open(args['write_file'], 'wb') as out_f:
    pass