#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables train_val_split,
batch_size as well as the transform function.
You are encouraged to modify these to improve the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

"""
   Answer to Question:

Briefly describe how your program works, and explain any design and training
decisions you made along the way.

The model is trained with GPU (it can be also trained with CPU but slower)

a. choice of architecture, algorithms and enhancements (if any)

    The main goal of the task is image classification. Convolutional neural networks is able to learn different features
and connect them to learn more complicated features. Several models such as Alexnet, VGG, and Resnet for image recognition 
tasks has been researched for this assignment.
    The final choice is a modified version of VGG net. The original VGG model is dealing with RGB pictures with size 
of 224*224. And for the assignment, the picture we are dealing with is grayscale with size of 64*64. The content of the 
picture is also simple cartoon characters with less details compared to picture of real people.Therefore, the size of 
the net has been reduced to avoid problems such as overfitting.
    The neural networks below can be divided to two parts. The first part is the convolutional layers. The layers have
similar patterns so a for loop is used to generates the convolutional layers. The second part is the fully connected
layers. The number of hidden nodes is reduced compare to the original VGG network because the pictures does not have
that much details and also the number of nodes in the linear layers increases the size of the model significantly.


b. choice of loss function and optimiser

    The loss function used is the standard nn.CrossEntropyLoss function which is usually used in image classification
tasks. The optimiser is Adam. Compare to SGD, Adam introduced momentum to speed up the training and amplify the speed of
escaping "rain gutter" shaped weight landscape so that the training would not oscillate without improvement.


c. choice of image transformations

    The image is first transformed to grayscale since the image is black and white. The second transform is horizontal
flip. It is reasonable to do this since the characters will face both left and right directions. 
    The rest of transforms are random_crop and random_rotation. The directions of the body and head are different when 
the character moves such as sitting down, keeping head down/up or lie down. Rotation is good enough to simulate these.
    The random_crop is also a important one. Sometimes part of the characters is blocked by objects. What object blocking
the body is not important but recognising the characters with parts missing is important.


d. tuning of metaparameters

    1. The batch size
     Smaller batch size is better than larger batch in some cases. The small batch will help the model escape the local
minimum which happens during training so I half the batch size. However, it takes more time to train the model but it is
a reasonable sacrifice. Plus small batch size can reduce the level of overfitting.
    2. The learning rate
     We can start with smaller learning rate and gradually increase it with the epoch according to the training result.
However, it is harder to do this since we cannot change anything in the hw2main.py. We choose a static learning rate of
0.002. Learning rate too high causing oscillation of the loss or the model not able to train successfully.
    3. Layers and number of hidden nodes
     The original model is VGG-16 layers model and it might be too complicated for the tasks. The training speed is too
low and the overfitting problem is serious. The saved model is higher than 300MB. The number of layers is reduced by one
and the number of out_channels is reduced too. The fully connected layer has significantly less number of hidden nodes 
to both reduce the size of the saved model and speed up training. 
    4. Number of epoch
     We cannot edit the hw2main.py in a way so that it can stop the epoch when the accuracy percentage reached the value
we satisfied or the loss is small enough. I left it with an estimated number of epoch based on the experience.
    5. Weight
     I also used the kaiming initialization on the weight to help the model train better. 


e. use of validation set, and any other steps taken to improve generalization and avoid overfitting

    In the training, the use of validation set shows there exists the problem of overfitting. The model behaves badly
when encounter unseen data. Several ways are introduced to improve the generalization including the previously mentioned
using smaller batch size.
    1. Batch normalization
     At first, the batch normalization is introduced to speed up the training. The original model takes 20 more epochs
to reach accuracy of 95%. The batch normalization regulates the output of the previous layers allowing for more stable
and efficient learning. Using the BN, the training got speed up and the accuracy on the training data is much closer to 
the accuracy on the validation data.   
    2. Dropout layers
     Several drop layers are added after the linear layers. The dropout layers randomly zeros some hidden nodes so that
the prediction of the model does not depend too much on specific, small features that does not decide the category of
the image.

"""


# vgg16_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing
    """
    if mode == 'train':
        train_transforms = transforms.Compose([

            # Make the picture grayscale
            transforms.Grayscale(1),

            # Flip the picture horizontally with a probability of 0.4
            transforms.RandomHorizontalFlip(0.4),

            # Rotate the picture between +20 -20 degree
            transforms.RandomRotation(degrees=20),

            # Crop the image randomly but keep the 64*64 size by add padding
            transforms.RandomCrop(64, padding=16),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        return train_transforms

    # We do not modify the test(validation) dataset
    elif mode == 'test':
        test_transforms = transforms.Compose([
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        return test_transforms


############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class Network(nn.Module):

    def __init__(self):
        super().__init__()

        # The image is black and white so in_channel=1
        # The forloop will construct the layers
        layers = []

        # The image is grayscale after the transform, so the image has one input channel
        in_channels = 1

        # This is a representation of the output channel or whether it is time to use maxpooling
        config = [32, 32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 256, 'M']
        for c in config:
            if c == 'M':

                # This represent the maxpooling is added to the layers variable
                layers += [nn.MaxPool2d(kernel_size=2)]
            else:

                # This is adding the convolutional layer
                # Then we use BN for generalization
                # Finally, we use ReLU as activation
                conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU()]

                # Update the in_channels of the next layer
                in_channels = c

        # When loop finishes, we will obtain the build of the convolutional layers according to the configuration
        self.conv = nn.Sequential(*layers)

        # Then, we build the fully connected layers
        # We add Dropout layer for generalization
        self.linear = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 140),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(140, 14),
        )

    def forward(self, t):
        t = self.conv(t)
        t = t.view(t.shape[0], -1)
        t = self.linear(t)
        t = F.log_softmax(t, dim=1)
        return t

# Weight initialization
def initialize_weight(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        nn.init.constant_(m.bias.data, 0)


net = Network()

# Apply the weight initialization
net = net.apply(initialize_weight)

# Use the standard Cross Entropy loss function
lossFunc = nn.CrossEntropyLoss()

############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 1
batch_size = 128
epochs = 130
optimiser = optim.Adam(net.parameters(), lr=0.002)
