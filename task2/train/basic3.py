import sys
from math import floor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
import cv2
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

DATADIR = '/data/datasets/tscherli/Data_Training_300/'
JSON = '/data/datasets/tscherli/goodlabels.json'
EPOCHS = 650
BATCH_SIZE = 16
LABEL_H = 864
LABEL_W = 1296
IMG_H = 256
IMG_W = 256
NCHANNELS = 1
TRAIN_RATIO = 0.1
PLOTFILE = "/data/datasets/tscherli/plot-256x-650e-3p3p3p3p-20.png"
MODELFILE = "/data/datasets/tscherli/model-256x-650e-3p3p3p3p-20.h5"

def pairwise(a):
  out = []
  for i in range(len(a)//2):
    x = a[2*i]
    y = a[2*i+1]
    out.append((x,y))
  return out

# read in dataset json file:
print('reading dataset labels:')
gt = pd.read_json(JSON)  # y is the training answers
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
    pairs = pairwise(label)
    normalizedpairs = []
    for x,y in pairs:
      normalizedpairs.append((x/LABEL_W, y/LABEL_H))
    label = []
    for (x,y) in normalizedpairs:
      label.append(x)
      label.append(y)
    labels.append(label)
    filenames.append(filename)
print('...done')

# define generator:
class MY_Generator(Sequence):

  def __init__(self, image_filenames, labels, batch_size):
    print("creating generator for",len(image_filenames),"images...")
    print('starting read:')

    #batches is a list of tuples: (batchx: numpy-array of images, batchy: numpy-array of labels)
    self.batches = []
    self.batch_size = batch_size
    self.nbatches = int(np.ceil(len(image_filenames)/batch_size))

    format_custom_text = progressbar.FormatCustomText('Batch: %(batch)d, Loaded: %(size).1f MB',dict(batch=0,size=0))
    bar = progressbar.ProgressBar(maxval=len(image_filenames),
      widgets=[progressbar.Bar('=', '[', ']'), ' ',
      progressbar.Percentage(), ' ',
      progressbar.AdaptiveETA(), ' ',
      progressbar.AnimatedMarker(markers='\|/-'),' ',
      format_custom_text])
    #bar.start()
    transferred = 0
    for b in range(self.nbatches):
      batchx = []
      batchy = []
      for i in range(batch_size):
        idx = b * batch_size + i
        if idx >= len(image_filenames): break
        filename = image_filenames[idx]
        label = labels[idx]
        img = cv2.resize(cv2.cvtColor(imread(DATADIR+filename), cv2.COLOR_BGR2GRAY), (IMG_H, IMG_W))
        img = img[...,np.newaxis]
        batchx.append(img)
        batchy.append(label)
        #bar.update(idx)
        transferred += sys.getsizeof(img)/1000000
        #format_custom_text.update_mapping(size=transferred)
      batchx = np.array(batchx)
      batchy = np.array(batchy)
      print("[",b,"of",self.nbatches,"] batch loaded. Batch Size:",sys.getsizeof(batchx)/1000000,"M. Total size:",transferred,"M.")
      format_custom_text.update_mapping(batch=b)
      self.batches.append((batchx, batchy))
    #bar.finish()

  def __len__(self):
    return self.nbatches

  def __getitem__(self, idx):
    return self.batches[idx]


# split data into test/train segments:
X_train, X_test, y_train, y_test = train_test_split(filenames, labels, test_size=TRAIN_RATIO)
NumTrain = len(X_train)
NumTest = len(X_test)
print("train/test split:",NumTrain,"/",NumTest)
generator_train = MY_Generator(X_train, y_train, BATCH_SIZE)
generator_test = MY_Generator(X_test, y_test, BATCH_SIZE)

print("Compiling network:")
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_W, IMG_H, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))
model.add(Flatten())
model.add(Dense(units=256, activation="relu"))
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=8, kernel_initializer="normal"))

K.tensorflow_backend._get_available_gpus()
parallelmodel = multi_gpu_model(model, gpus=4)

parallelmodel.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])


print("Fitting:")
H = parallelmodel.fit_generator(
    generator = generator_train,
    steps_per_epoch = len(generator_train),
    epochs=EPOCHS,
    verbose=2,
    validation_data = generator_test,
    validation_steps = len(generator_test),
    use_multiprocessing = True,
    workers = 1,
    max_queue_size = 1
    )

# save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")
parallelmodel.save(MODELFILE)
#
# # print("Evaluating:")
# # model.evaluate(X_test, y_test)
# # evaluate the network
# print("[INFO] evaluating network...")
# predictions = model.predict(X_test, batch_size=32)
# print(classification_report(y_test.argmax(axis=1),
#                             predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (256x-3p3p3p3p-20)")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(PLOTFILE)


print("k-fold cross validation:")
# k-fold cross validation

# kfold = KFold(n_splits=10)
# results = cross_val_score(model, data, labels, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# evaluate the network
# print("[INFO] evaluating network...")
# predictions = model.predict(data, batch_size=32)
# print(classification_report(labels.argmax(axis=1),
#                             predictions.argmax(axis=1), target_names=lb.classes_))
