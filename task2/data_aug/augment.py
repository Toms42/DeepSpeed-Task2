
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
from shapely.geometry import Polygon
from random import uniform

DATADIR = '/data/datasets/tscherli/Data_Training/'
JSON_IN = '/data/datasets/tscherli/goodlabels.json'
OUTDIR = '/data/datasets/tscherli/augmented/'
JSON_OUT = '/data/datasets/tscherli/augmentedlabels.json'
LABEL_H = 864
LABEL_W = 1296
IMG_H = 864
IMG_W = 1296
EXPLODEFACTOR = 20

DRAWLABELS = False


#Transformative parameters:
THETAMAX = 20
TRANSMAX = 0.2 * IMG_H
ISCALEMAX = 1.5
ISCALEMIN = 0.7
NISCALEMAX = 1.3
NISCALEMIN = 0.7
SHEERMAX= 0.3

#Color balance parameters
MINSATURATION = 0.5
MAXSATURATION = 2
MINHUE = -20
MAXHUE = 20
MINBRIGHTNESS = -40
MAXBRIGHTNESS = 40

def pairwise(a):
  out = []
  for i in range(len(a)//2):
    x = a[2*i]
    y = a[2*i+1]
    out.append((x,y))
  return out

def composeAffine(scale_iso, theta, trans, scale_noniso, sheer):
  afrow = np.array([0,0,1])
  rot = cv2.getRotationMatrix2D((IMG_H/2,IMG_W/2),theta,scale_iso)
  rot = np.vstack((rot,afrow))

  t = np.identity(3)
  t[0,2] = trans[0]
  t[1,2] = trans[1]

  sh = np.identity(3)
  sh[0,1] = sheer
  sh[0,2] = -sheer * IMG_W/2

  sq = np.identity(3)
  sq[0,0] = scale_noniso[0]
  sq[1,1] = scale_noniso[1]

  affine = np.matmul(np.matmul(np.matmul(t,sh),sq),rot)
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
  theta = uniform(-THETAMAX,THETAMAX)
  trans = (uniform(-TRANSMAX,TRANSMAX),uniform(-TRANSMAX,TRANSMAX))
  sheer = (uniform(-SHEERMAX,SHEERMAX))
  scale_iso = uniform(ISCALEMIN,ISCALEMAX)
  scale_noniso = (uniform(NISCALEMIN,NISCALEMAX),uniform(NISCALEMIN,NISCALEMAX))
  mat = composeAffine(scale_iso,theta,trans,scale_noniso,sheer)
  return applyAffine(img, label, mat)

def randomColor(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  img = np.asarray(img,dtype=int)

  hue = uniform(MINHUE,MAXHUE)
  brightness = uniform(MINBRIGHTNESS, MAXBRIGHTNESS)
  saturation = uniform(MINSATURATION, MAXSATURATION)

  img[:,:,0] = (img[:,:,0] + hue)
  img[:,:,1] = img[:,:,1] * saturation
  img[:,:,2] = img[:,:,2] + brightness
  img = np.clip(img,0,255)
  img = np.uint8(img)

  return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

###########################################
############ BEGIN SCRIPT #################
##########################################

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

generated=0
bar = progressbar.ProgressBar(maxval=len(filenames)*EXPLODEFACTOR,
  widgets=[progressbar.Bar('=', '[', ']'), ' ',
    progressbar.Percentage(), ' ',
    progressbar.AdaptiveETA(), ' ',
    progressbar.AnimatedMarker(markers='\|/-')])
#bar.start()
for i in range(len(filenames)):
  filename = filenames[i]
  label = labels[i]
  img = cv2.imread(DATADIR+filename)
  for j in range(EXPLODEFACTOR):
    destname = filename + "_variant" + str(j) + ".JPG"
    imgw,labelw = randomAffine(img,label)

    # Drop frames where gate is no longer in view
    pairs = pairwise(labelw)
    skip = False
    for (x,y) in pairs:
      if x < 0 or x >= LABEL_W or y < 0 or y >= LABEL_H:
        skip = True

    if not skip:
      imgw = randomColor(imgw)
      labelDict[destname] = [labelw]

    # Draw labels if directed:
    if DRAWLABELS:
      clippedpairs = []
      for x,y in pairs:
        clippedpairs.append((int(min(max(x,0),IMG_W-1)), int(min(max(y,0),IMG_H-1))))
      labelw = []
      for (x,y) in clippedpairs:
        labelw.append(x)
        labelw.append(y)
      cv2.line(imgw,(labelw[0],labelw[1]),(labelw[2],labelw[3]),(0,100,0),5)
      cv2.line(imgw,(labelw[2],labelw[3]),(labelw[4],labelw[5]),(0,150,0),5)
      cv2.line(imgw,(labelw[4],labelw[5]),(labelw[6],labelw[7]),(0,200,0),5)
      cv2.line(imgw,(labelw[6],labelw[7]),(labelw[0],labelw[1]),(0,255,0),5)

    # Write frame:
    if not skip:
      cv2.imwrite(OUTDIR + destname,imgw)
      generated+=1
    count+=1
    #bar.update(count)
  print("[",count,"of",EXPLODEFACTOR*len(filenames),"]")
#bar.finish()
with open(JSON_OUT, 'w') as fp:
  json.dump(labelDict, fp)

print("Generated",generated,"images:\n -JSON located at",JSON_OUT,"\n-Images located at",OUTDIR)
