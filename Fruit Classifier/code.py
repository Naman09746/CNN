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
