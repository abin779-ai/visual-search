import cv2
import glob
import numpy as np
from skimage.feature import hog
import math
from scipy.ndimage.measurements import label
from skimage.measure import regionprops

Hfeat1 = []
Hfeat2 = []
names = []
for i in glob.glob(r"C:\Users\Public\Documents\Python Scripts\OBJECT RECOGNITION\Visual search\bags\*.jpg"):
    names.append(i)
    img = cv2.imread(i)
    #cv2.waitKey(0)
    
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
        
    
    img1 = cv2.resize(img,(224,224))
    cv2.imshow("Result",img1)
    
    imgray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    
    Fhog,Ihog = hog(imgray,orientations=9,pixels_per_cell=(16,16),cells_per_block=(8,8), block_norm='L1', visualise=True, transform_sqrt=False, feature_vector=True, normalise=None)

    H1 = Fhog.transpose()
    Hfeat1.append(H1)

#    cv2.imshow('gray image',imgray)
#    cv2.waitKey(0)

    # binary image
    (thresh, imbw) = cv2.threshold(imgray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    imbw = cv2.bitwise_not(imbw)
#    cv2.imshow('bin1 image',imbw)
#    cv2.waitKey(0)
    
    kernel = np.ones((5,5),np.uint8)
    imclos = cv2.morphologyEx(imbw, cv2.MORPH_CLOSE, kernel)
     
#    cv2.imshow('bin2 image',imclos)
#    cv2.waitKey(0)


    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(imclos, connectivity=8)
    
    sizes = stats[1:, -1]; 
    nb_components = nb_components - 1
    
    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 200
    
    #your answer image
    img2 = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255 
    
    cv2.imshow('bin3 image',img2)
    cv2.waitKey(50)
    
    imlabel = label(img2)
    imlabel = imlabel[0]
    props = regionprops(imlabel)
    
    #centroid = props[0].centroid
    
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
    
    moments_hu = props[0].moments_hu
    
    #moments_normalized = props[0].moments_normalized
    
    hu = cv2.HuMoments(cv2.moments(img2)).flatten()
    
    orientation = props[0].orientation
    
    solidity = props[0].solidity


    feat = np.hstack((eccentricity,extent,asratio,moments_hu,hu,orientation,solidity))


    H2 = feat.transpose()
    Hfeat2.append(H2)
    
#Hfeat = np.delete(Hfeat,(0),axis=0)
  
np.savez('out5_bags',Hfeat1,Hfeat2,names)  

# h = np.load('outfile.npy')  

    