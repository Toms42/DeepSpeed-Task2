import sys
from math import floor
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import random
import progressbar
from pandas.io import pickle
from scipy.constants import lb
from skimage.color import rgb2gray
import json

DATADIR = '/data/datasets/tscherli/augmented-less/'
JSON_IN = '/data/datasets/tscherli/lessaugmentedlabels.json'
JSON_OUT = '/data/datasets/tscherli/lessaugmentedlabels-clean.json'

def pairwise(a):
  out = []
  for i in range(len(a)//2):
    x = a[2*i]
    y = a[2*i+1]
    out.append((x,y))
  return out

# read in dataset json file:
print('reading dataset labels:')
gt = pd.read_json(JSON_IN)  # y is the training answers
print('...done')
keys = list(gt.keys())
labels = {}

random.seed(412)
random.shuffle(keys)

print('Filtering and normalizing labels...')
bar = progressbar.ProgressBar(maxval=len(keys),
  widgets=[progressbar.Bar('=', '[', ']'), ' ',
  progressbar.Percentage(), ' ',
  progressbar.AdaptiveETA(), ' ',
  progressbar.AnimatedMarker(markers='\|/-')])
bar.start()
# filter out negatives and missing images:
dropped = 0
count = 0
for filename in keys:
  count+=1
  bar.update(count)
  if len(gt[filename][0]) == 0:
    dropped+=1
  elif not os.path.isfile(DATADIR + filename):
    dropped+=1
  else:
    label = gt[filename][0]
    labels[filename]=[label]
bar.finish()
with open(JSON_OUT, 'w') as fp:
  json.dump(labels, fp)

print("JSON Filtering completed. Dropped",dropped,"of",count,"files.")
