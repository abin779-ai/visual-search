from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
import cv2
import glob
import math

import os
# To calculate a normalized histogram 
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize
from skimage.feature import hog
 
model = ResNet50(weights='imagenet')

import numpy as np

from tkinter.filedialog import askopenfilename
img_path = askopenfilename()
im = cv2.imread(img_path)
height, width = im.shape[:2]
if height*width>350000:
   ht = int(height/3)
   wd = int(width/3)
   im = cv2.resize(im,(wd,ht))
#cv2.imshow("Input image",im)
#cv2.waitKey(0)

roi = cv2.selectROI(im)
     
img = im[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
ht, wd = img.shape[:2]   


    
if ht>wd:
    dif = ht-wd
    d1 = dif/2
    d1 = math.ceil(d1)
    im1 = np.ones((ht,d1,3),dtype=np.uint8)
    im1 = 255*im1
    img = np.hstack((im1,img,im1))  
else:
    dif = wd-ht
    d1 = dif/2
    d1 = math.ceil(d1)
    im1 = np.ones((d1,wd,3),dtype=np.uint8)
    im1 = 255*im1
    img = np.vstack((im1,img,im1)) 
    
#cv2.imshow("Input preprocessed",img)
#cv2.waitKey(0)
                
img1 = cv2.resize(img,(224,224))
imgray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = img1
x = np.expand_dims(img1, axis=0)
x = np.float32(x)
x = preprocess_input(x)

preds = model.predict(x)

print('Predicted:', decode_predictions(preds))


pmax = np.amax(preds)
pind = np.argmax(preds)
predsor = np.sort(preds)
indsor = np.argsort(preds)
p2ind = indsor[0,998]
p3ind = indsor[0,997]

p1 = [610,841,834,869]
p2 = [433,518,615,911,515,496,887,808,796,399,560]
# 549 envelope 911 wool 615 knee pad 796 ski-mask 399 abaya 560 futbal helmet

(imageID, label, probability) = decode_predictions(preds)[0][0]
    
# tshirt    
if pind in p1 or p2ind in p1 or p3ind in p1:
  label = 'T-SHIRT'  
  cv2.putText(img2,format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
  cv2.imshow("Input", img2)
  cv2.waitKey(0)

  Fhog,Ihog = hog(imgray,orientations=9,pixels_per_cell=(16,16),cells_per_block=(8,8), block_norm='L1', visualise=True, transform_sqrt=False, feature_vector=True, normalise=None)

  cv2.imshow("HOG",Ihog)
  cv2.waitKey(0)

  H1 = Fhog.transpose()

  X = np.load('out5_tshirt1.npy')  
  X = np.float32(X)

  r,c = X.shape[:2]
  D = np.zeros((1,r),dtype=np.float32)

  names = []
  for i in glob.glob(r"C:\Users\Public\Documents\Python Scripts\OBJECT RECOGNITION\deep-learning-models-master\tshirt1\*.jpg"):
     names.append(i)

  for j in range(0,r):    
      f = X[j,:]
      f = f.transpose()
      dist = np.linalg.norm(H1-f)   
      D[0,j] = dist
    
  Dsor = np.sort(D)
  Dint = np.argsort(D)
  D6 = Dint[0:5]

  import matplotlib.pyplot as plt
  img1 = cv2.resize(img,(224,224))
  I1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
  plt.imshow(I1)
  plt.axis('off')
  plt.title('Query Product')


  fig=plt.figure(figsize=(7,7))
  plt.title('SIMILAR PRODUCTS')
  plt.axis('off')
  columns = 2
  rows = 2
  for i in range(1, columns*rows+1):
    ii = Dint[0,i-1]
    I2 = cv2.imread(names[ii])
    I2 = cv2.cvtColor(I2,cv2.COLOR_BGR2RGB)
    fig.add_subplot(rows, columns, i)
    plt.axis('off')
    plt.imshow(I2)
    istr = str(i)
    plt.title(istr) 
    
    
# cap

elif pind in p2 or p2ind in p2 or p3ind in p2:
  label = 'CAP'  
  cv2.putText(img2,format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
  cv2.imshow("Input", img2)
  cv2.waitKey(0)

  Fhog,Ihog = hog(imgray,orientations=9,pixels_per_cell=(16,16),cells_per_block=(8,8), block_norm='L1', visualise=True, transform_sqrt=False, feature_vector=True, normalise=None)

  cv2.imshow("HOG",Ihog)
  cv2.waitKey(0)

  H1 = Fhog.transpose()

  X = np.load('out5_cap1.npy')  
  X = np.float32(X)

  r,c = X.shape[:2]
  D = np.zeros((1,r),dtype=np.float32)

  names = []
  for i in glob.glob(r"C:\Users\Public\Documents\Python Scripts\OBJECT RECOGNITION\deep-learning-models-master\cap1\*.jpg"):
     names.append(i)

  for j in range(0,r):    
      f = X[j,:]
      f = f.transpose()
      dist = np.linalg.norm(H1-f)   
      D[0,j] = dist
    
  Dsor = np.sort(D)
  Dint = np.argsort(D)
  D6 = Dint[0,0:5]

  import matplotlib.pyplot as plt
  img1 = cv2.resize(img,(224,224))
  I1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
  plt.imshow(I1)
  plt.axis('off')
  plt.title('Query Product')


  fig=plt.figure(figsize=(7,7))
  plt.title('SIMILAR PRODUCTS')
  plt.axis('off')
  columns = 2
  rows = 2
  for i in range(1, columns*rows+1):
    ii = Dint[0,i-1]
    I2 = cv2.imread(names[ii])
    I2 = cv2.cvtColor(I2,cv2.COLOR_BGR2RGB)
    fig.add_subplot(rows, columns, i)
    plt.axis('off')
    plt.imshow(I2)
    istr = str(i)
    plt.title(istr)   
  
    
else:
    lab = "product not available"   
    cv2.putText(img,format(lab), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    cv2.imshow("Object",img)    
    
    