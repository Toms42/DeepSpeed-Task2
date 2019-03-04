
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
import cv2
import random
import json
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
from shapely.geometry import Polygon

DATADIR = '/data/datasets/tscherli/Data_Training/'
JSON_IN = '/data/datasets/tscherli/goodlabels.json'
OUTDIR = '/data/datasets/tscherli/augmented/'
JSON_OUT = '/data/datasets/tscherli/augmentedlabels.json'
EPOCHS = 400
BATCH_SIZE = 16
LABEL_H = 864
LABEL_W = 1296
EXPLODEFACTOR = 1


MAXTHETA = 20
MAXTRANS = 0.3
MAXSCALE = 0.3
MAXSHEER = 0.3
CHANCEMIRROR = 0.3
CHANCEVERTMIRROR = 0.1
MAXSTOPS = 0.3
MAXCONTRAST = 0.2

def pairwise(a):
  out = []
  for i in range(len(a)//2):
    x = a[2*i]
    y = a[2*i+1]
    out.append((x,y))
  return out

def mirror(img, label):
  imgw = np.flip(img,0)
  pairs = pairwise(label)
  labelw = []
  for (x,y) in pairs:
    x = LABEL_W - x
    labelw.append(x)
    labelw.append(y)
  return (imgw, labelw)

def composeAffine(theta, trans, sheer):
  r = np.identity(3)
  c,s = np.cos(theta), np.sin(theta)
  r[0,0] = c
  r[1,0] = -s
  r[0,1] = s
  r[1,1] = c

  t = np.identity(3)
  t[0,2] = trans[0]
  t[1,2] = trans[1]

  s = np.identity(3)
  s[0,0] = sheer[0]
  s[1,1] = sheer[1]

  affine = np.matmul(np.matmul(s,t),r)
  affine = affine[0:2]
  return affine

def applyAffine(img, label, mat):
  imgw = cv2.warpAffine(img, mat, dsize=(img.shape[1],img.shape[0]))
  pairs = pairwise(label)
  labelw = []
  for (x,y) in pairs:
    h = np.array([x,y,1])
    newpair = np.matmul(mat,h)
    labelw.append(newpair[0])
    labelw.append(newpair[1])
  return (imgw, labelw)

def randomAffine(img, label):
  theta = math.pi/8
  trans = (0,0)
  sheer = (1,1)
  mat = composeAffine(theta,trans,sheer)
  return applyAffine(img, label, mat)

# read in dataset json file:
print('reading dataset labels:')
gt = pd.read_json(JSON_IN)  # y is the training answers
print('...done')
keys = list(gt.keys())
labels = []
filenames = []

random.seed(412)
random.shuffle(keys)

print('Filtering and normalizing labels...')
# filter out negatives and missing images:
for filename in keys:
  if len(gt[filename][0]) == 0:
    gt.drop(filename, axis=1, inplace=True)
  elif not os.path.isfile(DATADIR + filename):
    print("Bad File:", DATADIR + filename)
    gt.drop(filename, axis=1, inplace=True)
  else:
    label = gt[filename][0]
    labels.append(label)
    filenames.append(filename)
print('...done')


labelDict = {}
count = 0

bar = progressbar.ProgressBar(maxval=len(filenames)*EXPLODEFACTOR,
  widgets=[progressbar.Bar('=', '[', ']'), ' ',
    progressbar.Percentage(), ' ',
    progressbar.AdaptiveETA(), ' ',
    progressbar.AnimatedMarker(markers='\|/-')])
bar.start()
for i in range(len(filenames)):
  filename = filenames[i]
  label = labels[i]
  img = cv2.imread(DATADIR+filename)
  for j in range(EXPLODEFACTOR):
    destname = filename + "_variant" + str(j) + ".JPG"
    imgw,labelw = randomAffine(img,label)
    labelDict[destname] = [labelw]
    cv2.imwrite(OUTDIR + destname,imgw)
    count+=1
    bar.update(count)
bar.finish()
with open(JSON_OUT, 'w') as fp:
  json.dump(labelDict, fp)

print("completed:\n -JSON located at",JSON_OUT,"\n-Images located at",OUTDIR)
