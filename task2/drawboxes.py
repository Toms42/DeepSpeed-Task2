import sys
from math import floor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import random
import progressbar
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
import glob

IMGDIR = '/data/datasets/tscherli/augmented/'
OUTDIR = '/data/datasets/tscherli/Outdir/Augmented-drawn/'
JSON_PRED = '/data/datasets/tscherli/augmentedlabels.json'
JSON_TRUTH = '/data/datasets/tscherli/training_GT_labels_v2.json'
NOGT = True
IMGH = 864
IMGW = 1296

def pairwise(a):
  out = []
  for i in range(len(a)//2):
    x = a[2*i]
    y = a[2*i+1]
    out.append((x,y))
  return out


print("reading ground-truth labels...")
truth = pd.read_json(JSON_TRUTH)  # y is the training answers

print("reading predicted labels...")
pred = pd.read_json(JSON_PRED)  # y is the training answers

filenames = pred.keys()

print("Running for",len(filenames),"files...");

bar = progressbar.ProgressBar(maxval=len(filenames),
    widgets=[progressbar.Bar('=', '[', ']'), ' ',
    progressbar.Percentage(), ' ',
    progressbar.AdaptiveETA()])
count = 0
bar.start()

for filename in filenames:
  count+=1
  bar.update(count)

  labels = pred[filename][0]
  pairs = pairwise(labels)
  clippedpairs = []
  for x,y in pairs:
    clippedpairs.append((int(min(max(x,0),IMGW-1)), int(min(max(y,0),IMGH-1))))
  labels = []
  for (x,y) in clippedpairs:
    labels.append(x)
    labels.append(y)

  image_with_rectangle = cv2.imread(IMGDIR+filename)
  cv2.line(image_with_rectangle,(labels[0],labels[1]),(labels[2],labels[3]),(0,100,0),5)
  cv2.line(image_with_rectangle,(labels[2],labels[3]),(labels[4],labels[5]),(0,150,0),5)
  cv2.line(image_with_rectangle,(labels[4],labels[5]),(labels[6],labels[7]),(0,200,0),5)
  cv2.line(image_with_rectangle,(labels[6],labels[7]),(labels[0],labels[1]),(0,255,0),5)

  if not NOGT:
    labels = truth[filename][0]
    pairs = pairwise(labels)
    clippedpairs = []
    for x,y in pairs:
      clippedpairs.append((int(min(max(x,0),IMGW-1)), int(min(max(y,0),IMGH-1))))
    labels = []
    for (x,y) in clippedpairs:
      labels.append(x)
      labels.append(y)

    cv2.line(image_with_rectangle,(labels[0],labels[1]),(labels[2],labels[3]),(150,0,0),5)
    cv2.line(image_with_rectangle,(labels[2],labels[3]),(labels[4],labels[5]),(255,0,0),5)
    cv2.line(image_with_rectangle,(labels[4],labels[5]),(labels[6],labels[7]),(255,0,0),5)
    cv2.line(image_with_rectangle,(labels[6],labels[7]),(labels[0],labels[1]),(255,0,0),5)

  cv2.imwrite(OUTDIR + filename + "_visualized.JPG",image_with_rectangle)
bar.finish()

print("done! Files written into ",OUTDIR)
