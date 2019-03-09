# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:38:13 2017

@author: ABABRAHA
"""

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
import cv2
 
model = ResNet50(weights='imagenet')

import numpy as np

cap = cv2.VideoCapture(0)
#cap.set(3,1080)
#cap.set(4,720)

while(True):

    ret, img = cap.read()
    img1 = cv2.resize(img,(224,224))
    x = np.expand_dims(img1, axis=0)
    x = np.float32(x)
    x = preprocess_input(x)


    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    
    pmax = np.amax(preds)
    pind = np.argmax(preds)

    print('Predicted:', decode_predictions(preds))
    # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]

    (imageID, label, probability) = decode_predictions(preds)[0][0]

    pall = [409,414,418,447,453,454,455,464,487,502,504,508,514,518,526,527,528,530,531,532,545,560,590,591,608,610,620,630,636,655,664,673,681,689,697,709,748,770,774,782,806,826,834,836,837,841,851,879,892,893,898,921]

    if pind in pall and pmax>.1:
    
        cv2.putText(img,format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow("Object", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    else:
        lab = "product not in list"   
        cv2.putText(img,format(lab), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.imshow("Object", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()

