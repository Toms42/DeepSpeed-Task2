import cv2
import os
import progressbar

DIR = 'augmented_300/'
DIR_IN = 'augmented/'



count=0
print("Getting directory listing:")
listing = os.walk(DIR_IN)

print("Getting file count:")
for root, dir, files in listing:
  for f in files:
     count+=1

print("Resizing",count,"files:")
bar = progressbar.ProgressBar(maxval=count,
    widgets=[progressbar.Bar('=', '[', ']'), ' ',
    progressbar.Percentage(), ' ',
    progressbar.AdaptiveETA()])
count = 0
bar.start()
for root, dir, files in listing:
  for f in files:
     count+=1
     bar.update(count)
     im = cv2.imread(f)
     new_im = cv2.resize(im, None, fx=0.5, fy=0.5)
     cv2.imwrite(new_im, DIR + f)
bar.finish()
