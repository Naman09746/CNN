

# importing
import random
import pandas as pd
import numpy as np
import torchvision
import torch
from torch import nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# Loading data
transforms_train = T.Compose([T.ToTensor(),T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
image_data_train = ImageFolder("./fruits-360/Training",transform=transforms_train)
image_data_test = ImageFolder("./fruits-360/Validation",transform=transforms_train)

# Shuffling data and then collecting all the labels.
random.shuffle(image_data_train.samples)
random.shuffle(image_data_test.samples)

# Total classes
classes_idx = image_data_train.class_to_idx
classes = len(image_data_train.classes)
len_train_data = len(image_data_train)
len_test_data = len(image_data_test)


def get_labels():
    labels_train = [] # All the labels
    labels_test = []
    for i in image_data_train.imgs:
        labels_train.append(i[1])
    
    for j in image_data_test.imgs:
        labels_test.append(j[1])
    
    return (labels_train, labels_test)

labels_train, labels_test = get_labels()
