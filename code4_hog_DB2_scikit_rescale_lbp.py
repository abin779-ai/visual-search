from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
import cv2
import glob
import math

import os
# Local Binary Pattern function
from skimage.feature import local_binary_pattern
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
I = img1
x = np.expand_dims(img1, axis=0)
x = np.float32(x)
x = preprocess_input(x)


preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds))
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]

(imageID, label, probability) = decode_predictions(preds)[0][0]

cv2.putText(img,format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
cv2.imshow("Input", img)
cv2.waitKey(0)

imgray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

Fhog,Ihog = hog(imgray,orientations=9,pixels_per_cell=(16,16),cells_per_block=(8,8), block_norm='L1', visualise=True, transform_sqrt=False, feature_vector=True, normalise=None)

cv2.imshow("HOG",Ihog)
cv2.waitKey(0)

H1 = Fhog.transpose()

# LBP feature extraction
P = 8
R = 3   
lbp = local_binary_pattern(imgray, P, R)
Ilbp = np.uint8(lbp)
cv2.imshow("LBP",Ilbp)
cv2.waitKey(0)
    
bins = int(lbp.max() + 1)
hist,_= np.histogram(lbp, normed=True, bins=bins, range=(0,bins))
H2 = hist.transpose()


X1 = np.load('out5_tshirt1.npy')  
X1 = np.float32(X1)

X2 = np.load('out6_tshirt1.npy')  
X2 = np.float32(X2)

r1,c1 = X1.shape[:2]
r2,c2 = X2.shape[:2]

D1 = np.zeros((1,r1),dtype=np.float32)
D2 = np.zeros((1,r2),dtype=np.float32)


#import scipy
names = []
for i in glob.glob(r"C:\Users\Public\Documents\Python Scripts\OBJECT RECOGNITION\deep-learning-models-master\tshirt1\*.jpg"):
    names.append(i)

for j in range(0,r1):
    
    f1 = X1[j,:]
    f1 = f1.transpose()
    dist = np.linalg.norm(H1-f1)  
    D1[0,j] = dist
    
    f2 = X2[j,:]
    f2 = f2.transpose()
    dist = np.linalg.norm(H2-f2)  
    D2[0,j] = dist
    
    
D = D1+(D2*5)
Dsor = np.sort(D)
Dint = np.argsort(D)


D6 = Dint[0:5]


import matplotlib.pyplot as plt

I1 = cv2.cvtColor(I,cv2.COLOR_BGR2RGB)
plt.imshow(I1)
plt.axis('off')
plt.title('Query Product')


fig=plt.figure(figsize=(7,7))
plt.title('SIMILAR PRODUCTS')
plt.axis('off')
columns = 2
rows = 2
for i in range(1, columns*rows+1):
    ii = Dint[0,i]
    I = cv2.imread(names[ii])
    I = cv2.cvtColor(I,cv2.COLOR_BGR2RGB)
    fig.add_subplot(rows, columns, i)
    plt.axis('off')
    plt.imshow(I)
    istr = str(i)
    plt.title(istr)
#    plt.xlabel(istr)
#    plt.axis('off')
