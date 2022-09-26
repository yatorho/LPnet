# Project report of license plate recognition based on CRNN and CTC

Our group members are: 王威虎、杨世博、杨镇宇、邓京雨

## Contents

- [Project report of license plate recognition based on CRNN and CTC](#project-report-of-license-plate-recognition-based-on-crnn-and-ctc)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Data](#data)
  - [Model](#model)
  - [Training](#training)
  - [Conclusion](#conclusion)
  - [Reference](#reference)

## Introduction 

The CRNN(Convolutional Recurrent Neural Network) is a deep learning model that combines the advantages of CNN(Convolutional Neural Network) and RNN(Recurrent Neural Network). It is widely used in the field of image recognition. The CTC(Connectionist Temporal Classification) is a loss function that can be used to train the RNN model. It is widely used in the field of sequence recognition. In this project, we use the CNN model to extract the features of the license plate image and select the area of interest, and then use the RNN model to recognize the characters in the license plate image. Finally, we use the CTC loss function to train the model.

Alphabets of CRNN model is a list of characters that can be recognized by the model. The length of the list is 68, including 26 lowercase letters, 26 uppercase letters, 10 numbers, and 6 special characters. The length of the license plate is 7, so the length of the input of the CRNN model is 7. The length of the output of the CRNN model is 68. The output of the CRNN model is a 7*68 matrix, and the value of each element in the matrix represents the probability of the corresponding character. The character with the highest probability in each column is the character recognized by the model.

alphabets = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

y = "hello" == labels = [41, 38, 45, 45, 49]

## Data

The data set used in this project is the CCPD dataset, which contains more than 300,000 license plate images.

However, the CCPD dataset is not a standard dataset, and most of photos are token in Anhui Province, China, which causes the license plate characters starts with "皖". So we need to clean the data set. We use the following steps to clean the data set:

## Model

The model we use is CRNN, which is a combination of CNN and RNN. The model structure is shown in the following figure: => We need a picture here!

The model consists of three parts: CNN, RNN and CTC. The CNN part is used to extract the features of the license plate image, and the RNN part is used to recognize the characters in the license plate image. The CTC part is used to train the model.

We use a model like ResNet50 to extract the AOI(Area of Interest) of the license plate image. The input of the model is a 3-channel image,
and the output is a tensor with shape(?, 4). The 4 elements in the tensor represent the coordinates of the AOI. The coordinates are normalized to the range of 0 to 1. The coordinates are in the order of left, top, right, bottom.

The CNN in CRNN model we use is like VGG16, which consists of 10 convolutional layers and 3 fully connected layers. We use a nn.Linear(512, 4) as the last layer to extract the region of license plate.

We use the LSTM model as the RNN part. The LSTM model is a special RNN model, which can solve the problem of gradient vanishing and gradient explosion. The LSTM model consists of three gates: input gate, forget gate and output gate. The input gate controls the input of the cell state, the forget gate controls the output of the cell state, and the output gate controls the output of the hidden state.

The CTC part is used to train the model. The CTC part consists of two parts: the CTC loss function and the CTC decoder. The CTC loss function is used to calculate the loss of the model, and the CTC decoder is used to decode the output of the RNN model.

## Training

We use the Adam optimizer to train the model. The learning rate is 0.001, and the batch size is 24. We train the model for 15 epochs. 

For demonstration, we just train the model with 20000 images.

We would like to train two sub_model separately with the same hyperparameters. The first sub_model is trained with the original data set, and the second sub_model is trained with the cleaned data set.

## Conclusion

## Reference

[1] [CRNN: An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717)

[2] [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/icml_2006.pdf)

[3] [CCPD: A Large-scale Chinese Commercial License Plate Dataset](https://arxiv.org/abs/1904.01906)

[4] [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

[5] [VGGNet](https://arxiv.org/abs/1409.1556)

[6] [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)

[7] [LSTM: A Search Space Odyssey](https://arxiv.org/abs/1503.04069)

[8] [CTC Decoder](https://distill.pub/2017/ctc/)

[9] [CTC Loss](https://distill.pub/2017/ctc/)

[10] [Pytorch](https://pytorch.org/)
