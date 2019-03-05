import sys
import cv2
import os
import multiprocessing

#usage: python3 script.py <source> <dir>

DIR_IN = sys.argv[1]
DIR_OUT = sys.argv[2]

def resize_image(root, dirname, files):
    print(root, dirname, files)
    for f in files:
        im = cv2.imread(os.sep.join([root, f]))
        new_im = cv2.resize(im, None, fx=0.5, fy=0.5)
        cv2.imwrite(os.sep.join([DIR_OUT, f]), new_im)
    return True

pool = multiprocessing.Pool()
pool.starmap(resize_image, os.walk(DIR_IN))
