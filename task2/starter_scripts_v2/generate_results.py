# This script is to be filled by the team members.
# Import necessary libraries
# Load libraries
import json
import cv2
import numpy as np
import sys
from math import floor
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import progressbar
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.utils import multi_gpu_model
from keras.utils import Sequence
from pandas.io import pickle
from scipy.constants import lb
from skimage.color import rgb2gray
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from skimage.io import imread
from skimage.transform import resize
from keras import backend as K
from keras.models import load_model

# Implement a function that takes an image as an input, performs any preprocessing steps and outputs a list of bounding box detections and assosciated confidence score.

IMG_H = 300
IMG_W = 400
LABEL_H = 864
LABEL_W = 1296
MODELFILE = "/data/datasets/tscherli/models/model-300x-650e-133p33p-30.h5"

def pairwise(a):
  out = []
  for i in range(len(a)//2):
    x = a[2*i]
    y = a[2*i+1]
    out.append((x,y))
  return out

class GenerateFinalDetections():
    def __init__(self):
        print("loading model:")
        K.clear_session()
        self.model = load_model(MODELFILE)
        self.model.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])

    def predict(self,img):
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),(IMG_H, IMG_W))
        img = img[np.newaxis,...,np.newaxis]
        label = self.model.predict(img).tolist()
        label2 = [max(min(l,1),0) for l in label[0]]
        pairs = pairwise(label2)
        normalizedpairs = []
        for x,y in pairs:
          normalizedpairs.append((x*LABEL_W, y*LABEL_H))
        label = []
        for (x,y) in normalizedpairs:
          label.append(x)
          label.append(y)
        labels = [label]
        return labels
