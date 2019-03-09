from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
import cv2
import glob
import math
from scipy.ndimage.measurements import label
from skimage.measure import regionprops
import matplotlib.pyplot as plt
 
import os
# Local Binary Pattern function
from skimage.feature import local_binary_pattern
# To calculate a normalized histogram 
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize
from skimage.feature import hog
 
plt.close('all')
cv2.destroyAllWindows()
model = ResNet50(weights='imagenet')

import numpy as np


from tkinter.filedialog import askopenfilename
img_path = askopenfilename()
img = cv2.imread(img_path)
ht, wd = img.shape[:2]   

#cv2.imshow("Input image",img)
#cv2.waitKey(0)
    

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
img2 = img1
x = np.expand_dims(img1, axis=0)
x = np.float32(x)
x = preprocess_input(x)


preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds))
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]


pmax = np.amax(preds)
pind = np.argmax(preds)
predsor = np.sort(preds)
indsor = np.argsort(preds)
p2ind = indsor[0,998]
p3ind = indsor[0,997]



p1 = [610,841,834,869]
p2 = [433,518,615,911,515,496,887,808,796,399,560]
p3 = [414,636,797,748]
p4 = [836,837]
p5 = [409,531,530,600,826]


(imageID, label1, probability) = decode_predictions(preds)[0][0]
cv2.putText(img,format(label1), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
cv2.imshow("Prediction", img)
cv2.waitKey(100)

imgray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

# tshirt

if pind in p1 or p2ind in p1:
    label2 = 'T-SHIRT'  
    cv2.putText(img2,format(label2), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    cv2.imshow("Input", img2)
    cv2.waitKey(100)

    Fhog,Ihog = hog(imgray,orientations=9,pixels_per_cell=(16,16),cells_per_block=(8,8), block_norm='L1', visualise=True, transform_sqrt=False, feature_vector=True, normalise=None)
    
    cv2.imshow("HOG",Ihog)
    cv2.waitKey(100)
    
    H1 = Fhog.transpose()
    
    # shape features----------------------------------------------
    (thresh, imbw) = cv2.threshold(imgray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    imbw = cv2.bitwise_not(imbw)
        
    kernel = np.ones((5,5),np.uint8)
    imclos = cv2.morphologyEx(imbw, cv2.MORPH_CLOSE, kernel)
    
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(imclos, connectivity=8)
        
    sizes = stats[1:, -1]; 
    nb_components = nb_components - 1
        
    min_size = 200
        
    img2 = np.zeros((output.shape))
    for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = 255 
    
    cv2.imshow('bin3 image',img2)
    cv2.waitKey(100)
        
    imlabel = label(img2)
    imlabel = imlabel[0]
    props = regionprops(imlabel) 
    area = props[0].area  
    convex_area = props[0].convex_area 
    eccentricity = props[0].eccentricity  
    extent = props[0].extent   
    major_axis_length = props[0].major_axis_length 
    minor_axis_length = props[0].minor_axis_length   
    if major_axis_length==0:
       asratio=0
    else:
       asratio = minor_axis_length/major_axis_length
    asratio = minor_axis_length/major_axis_length  
    moments_hu = props[0].moments_hu    
    hu = cv2.HuMoments(cv2.moments(img2)).flatten()
    orientation = props[0].orientation   
    solidity = props[0].solidity
    
    feat = np.hstack((eccentricity,extent,asratio,moments_hu,hu,orientation,solidity))
    H2 = feat.transpose()

    npzfile = np.load('out5_tshirt1.npz')
    
    X1 = npzfile['arr_0']
    X1 = np.float32(X1)
    
    X2 = npzfile['arr_1'] 
    X2 = np.float32(X2)
    
    r1,c1 = X1.shape[:2]
    r2,c2 = X2.shape[:2]
    
    D1 = np.zeros((1,r1),dtype=np.float32)
    D2 = np.zeros((1,r2),dtype=np.float32)
    
    names = npzfile['arr_2']
    names = list(names)
    
    for j in range(0,r1):
        
        f1 = X1[j,:]
        f1 = f1.transpose()
        dist = np.linalg.norm(H1-f1)  
        D1[0,j] = dist
        
        f2 = X2[j,:]
        f2 = f2.transpose()
        dist = np.linalg.norm(H2-f2)  
        D2[0,j] = dist
        
        
    D = D1+D2
    Dsor = np.sort(D)
    Dint = np.argsort(D)
    D6 = Dint[0:5]
    
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
     
    
# bag

elif pind in p3 or p2ind in p3:
    label2 = 'BAG'  
    cv2.putText(img2,format(label2), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    cv2.imshow("Input", img2)
    cv2.waitKey(100)

    Fhog,Ihog = hog(imgray,orientations=9,pixels_per_cell=(16,16),cells_per_block=(8,8), block_norm='L1', visualise=True, transform_sqrt=False, feature_vector=True, normalise=None)
    
    cv2.imshow("HOG",Ihog)
    cv2.waitKey(100)
    
    H1 = Fhog.transpose()
    
    # shape features----------------------------------------------
    (thresh, imbw) = cv2.threshold(imgray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    imbw = cv2.bitwise_not(imbw)
        
    kernel = np.ones((5,5),np.uint8)
    imclos = cv2.morphologyEx(imbw, cv2.MORPH_CLOSE, kernel)
    
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(imclos, connectivity=8)
        
    sizes = stats[1:, -1]; 
    nb_components = nb_components - 1
        
    min_size = 200
        
    img2 = np.zeros((output.shape))
    for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = 255 
                
    cv2.imshow('bin3 image',img2)
    cv2.waitKey(100)
        
    imlabel = label(img2)
    imlabel = imlabel[0]
    props = regionprops(imlabel) 
    area = props[0].area  
    convex_area = props[0].convex_area 
    eccentricity = props[0].eccentricity  
    extent = props[0].extent   
    major_axis_length = props[0].major_axis_length 
    minor_axis_length = props[0].minor_axis_length   
    if major_axis_length==0:
       asratio=0
    else:
       asratio = minor_axis_length/major_axis_length
    asratio = minor_axis_length/major_axis_length  
    moments_hu = props[0].moments_hu    
    hu = cv2.HuMoments(cv2.moments(img2)).flatten()
    orientation = props[0].orientation   
    solidity = props[0].solidity
    
    feat = np.hstack((eccentricity,extent,asratio,moments_hu,hu,orientation,solidity))
    H2 = feat.transpose()

    npzfile = np.load('out5_bags1.npz')
    
    X1 = npzfile['arr_0']
    X1 = np.float32(X1)
    
    X2 = npzfile['arr_1'] 
    X2 = np.float32(X2)
    
    r1,c1 = X1.shape[:2]
    r2,c2 = X2.shape[:2]
    
    D1 = np.zeros((1,r1),dtype=np.float32)
    D2 = np.zeros((1,r2),dtype=np.float32)
    
    #import scipy
    names = npzfile['arr_2']
    names = list(names)
    
    for j in range(0,r1):
        
        f1 = X1[j,:]
        f1 = f1.transpose()
        dist = np.linalg.norm(H1-f1)  
        D1[0,j] = dist
        
        f2 = X2[j,:]
        f2 = f2.transpose()
        dist = np.linalg.norm(H2-f2)  
        D2[0,j] = dist
        
        
    D = D1+D2
    Dsor = np.sort(D)
    Dint = np.argsort(D)
    D6 = Dint[0:5]
    
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
    
    

# sunglass

elif pind in p4 or p2ind in p4:
    label2 = 'SUNGLASS'  
    cv2.putText(img2,format(label2), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    cv2.imshow("Input", img2)
    cv2.waitKey(100)

    Fhog,Ihog = hog(imgray,orientations=9,pixels_per_cell=(16,16),cells_per_block=(8,8), block_norm='L1', visualise=True, transform_sqrt=False, feature_vector=True, normalise=None)
    
    cv2.imshow("HOG",Ihog)
    cv2.waitKey(100)
    
    H1 = Fhog.transpose()
    
    # shape features----------------------------------------------
    (thresh, imbw) = cv2.threshold(imgray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    imbw = cv2.bitwise_not(imbw)
        
    kernel = np.ones((5,5),np.uint8)
    imclos = cv2.morphologyEx(imbw, cv2.MORPH_CLOSE, kernel)
    
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(imclos, connectivity=8)
        
    sizes = stats[1:, -1]; 
    nb_components = nb_components - 1
        
    min_size = 200
        
    img2 = np.zeros((output.shape))
    for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = 255 
                
    cv2.imshow('bin3 image',img2)
    cv2.waitKey(100)
        
    imlabel = label(img2)
    imlabel = imlabel[0]
    props = regionprops(imlabel) 
    area = props[0].area  
    convex_area = props[0].convex_area 
    eccentricity = props[0].eccentricity  
    extent = props[0].extent   
    major_axis_length = props[0].major_axis_length 
    minor_axis_length = props[0].minor_axis_length   
    if major_axis_length==0:
       asratio=0
    else:
       asratio = minor_axis_length/major_axis_length
    asratio = minor_axis_length/major_axis_length  
    moments_hu = props[0].moments_hu    
    hu = cv2.HuMoments(cv2.moments(img2)).flatten()
    orientation = props[0].orientation   
    solidity = props[0].solidity
    
    feat = np.hstack((eccentricity,extent,asratio,moments_hu,hu,orientation,solidity))
    H2 = feat.transpose()

    npzfile = np.load('out5_sunglasses1.npz')
    
    X1 = npzfile['arr_0']
    X1 = np.float32(X1)
    
    X2 = npzfile['arr_1'] 
    X2 = np.float32(X2)
    
    r1,c1 = X1.shape[:2]
    r2,c2 = X2.shape[:2]
    
    D1 = np.zeros((1,r1),dtype=np.float32)
    D2 = np.zeros((1,r2),dtype=np.float32)
    
    #import scipy
    names = npzfile['arr_2']
    names = list(names)
    
    for j in range(0,r1):
        
        f1 = X1[j,:]
        f1 = f1.transpose()
        dist = np.linalg.norm(H1-f1)  
        D1[0,j] = dist
        
        f2 = X2[j,:]
        f2 = f2.transpose()
        dist = np.linalg.norm(H2-f2)  
        D2[0,j] = dist
        
        
    D = D1+D2
    Dsor = np.sort(D)
    Dint = np.argsort(D)
    D6 = Dint[0:5]
    
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
    
    
    
# watch

elif pind in p5 or p2ind in p5:
    label2 = 'WATCH'  
    cv2.putText(img2,format(label2), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    cv2.imshow("Input", img2)
    cv2.waitKey(100)

    Fhog,Ihog = hog(imgray,orientations=9,pixels_per_cell=(16,16),cells_per_block=(8,8), block_norm='L1', visualise=True, transform_sqrt=False, feature_vector=True, normalise=None)
    
    cv2.imshow("HOG",Ihog)
    cv2.waitKey(100)
    
    H1 = Fhog.transpose()
    
    # shape features----------------------------------------------
    (thresh, imbw) = cv2.threshold(imgray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    imbw = cv2.bitwise_not(imbw)
        
    kernel = np.ones((5,5),np.uint8)
    imclos = cv2.morphologyEx(imbw, cv2.MORPH_CLOSE, kernel)
    
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(imclos, connectivity=8)
        
    sizes = stats[1:, -1]; 
    nb_components = nb_components - 1
        
    min_size = 200
        
    img2 = np.zeros((output.shape))
    for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = 255 
                
    cv2.imshow('bin3 image',img2)
    cv2.waitKey(100)
        
    imlabel = label(img2)
    imlabel = imlabel[0]
    props = regionprops(imlabel) 
    area = props[0].area  
    convex_area = props[0].convex_area 
    eccentricity = props[0].eccentricity  
    extent = props[0].extent   
    major_axis_length = props[0].major_axis_length 
    minor_axis_length = props[0].minor_axis_length   
    if major_axis_length==0:
       asratio=0
    else:
       asratio = minor_axis_length/major_axis_length
    asratio = minor_axis_length/major_axis_length  
    moments_hu = props[0].moments_hu    
    hu = cv2.HuMoments(cv2.moments(img2)).flatten()
    orientation = props[0].orientation   
    solidity = props[0].solidity
    
    feat = np.hstack((eccentricity,extent,asratio,moments_hu,hu,orientation,solidity))
    H2 = feat.transpose()

    npzfile = np.load('out5_watchnew1.npz')
    
    X1 = npzfile['arr_0']
    X1 = np.float32(X1)
    
    X2 = npzfile['arr_1'] 
    X2 = np.float32(X2)
    
    r1,c1 = X1.shape[:2]
    r2,c2 = X2.shape[:2]
    
    D1 = np.zeros((1,r1),dtype=np.float32)
    D2 = np.zeros((1,r2),dtype=np.float32)
    
    #import scipy
    names = npzfile['arr_2']
    names = list(names)
    
    for j in range(0,r1):
        
        f1 = X1[j,:]
        f1 = f1.transpose()
        dist = np.linalg.norm(H1-f1)  
        D1[0,j] = dist
        
        f2 = X2[j,:]
        f2 = f2.transpose()
        dist = np.linalg.norm(H2-f2)  
        D2[0,j] = dist
        
        
    D = D1+D2
    Dsor = np.sort(D)
    Dint = np.argsort(D)
    D6 = Dint[0:5]
    
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
    
    
# cap

elif pind in p2 or p2ind in p2:
    label2 = 'CAP'  
    cv2.putText(img2,format(label2), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    cv2.imshow("Input", img2)
    cv2.waitKey(100)

    Fhog,Ihog = hog(imgray,orientations=9,pixels_per_cell=(16,16),cells_per_block=(8,8), block_norm='L1', visualise=True, transform_sqrt=False, feature_vector=True, normalise=None)
    
    cv2.imshow("HOG",Ihog)
    cv2.waitKey(100)
    
    H1 = Fhog.transpose()
    
    # shape features----------------------------------------------
    (thresh, imbw) = cv2.threshold(imgray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    imbw = cv2.bitwise_not(imbw)
        
    kernel = np.ones((5,5),np.uint8)
    imclos = cv2.morphologyEx(imbw, cv2.MORPH_CLOSE, kernel)
    
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(imclos, connectivity=8)
        
    sizes = stats[1:, -1]; 
    nb_components = nb_components - 1
        
    min_size = 200
        
    img2 = np.zeros((output.shape))
    for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = 255 
                
    cv2.imshow('bin3 image',img2)
    cv2.waitKey(100)
        
    imlabel = label(img2)
    imlabel = imlabel[0]
    props = regionprops(imlabel) 
    area = props[0].area  
    convex_area = props[0].convex_area 
    eccentricity = props[0].eccentricity  
    extent = props[0].extent   
    major_axis_length = props[0].major_axis_length 
    minor_axis_length = props[0].minor_axis_length   
    if major_axis_length==0:
       asratio=0
    else:
       asratio = minor_axis_length/major_axis_length
    asratio = minor_axis_length/major_axis_length  
    moments_hu = props[0].moments_hu    
    hu = cv2.HuMoments(cv2.moments(img2)).flatten()
    orientation = props[0].orientation   
    solidity = props[0].solidity
    
    feat = np.hstack((eccentricity,extent,asratio,moments_hu,hu,orientation,solidity))
    H2 = feat.transpose()

    npzfile = np.load('out5_capsandhats1.npz')
    
    X1 = npzfile['arr_0']
    X1 = np.float32(X1)
    
    X2 = npzfile['arr_1'] 
    X2 = np.float32(X2)
    
    r1,c1 = X1.shape[:2]
    r2,c2 = X2.shape[:2]
    
    D1 = np.zeros((1,r1),dtype=np.float32)
    D2 = np.zeros((1,r2),dtype=np.float32)
    
    #import scipy
    names = npzfile['arr_2']
    names = list(names)
    
    for j in range(0,r1):
        
        f1 = X1[j,:]
        f1 = f1.transpose()
        dist = np.linalg.norm(H1-f1)  
        D1[0,j] = dist
        
        f2 = X2[j,:]
        f2 = f2.transpose()
        dist = np.linalg.norm(H2-f2)  
        D2[0,j] = dist
        
        
    D = D1+D2
    Dsor = np.sort(D)
    Dint = np.argsort(D)
    D6 = Dint[0:5]
    
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
    
else:
    label2 = "product not available"   
    cv2.putText(img2,format(label2), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    cv2.imshow("Object",img2)    
    