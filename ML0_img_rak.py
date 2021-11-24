# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 22:47:28 2021
@author: E Bonnet
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import IPython.display as display

ksize = 150  #Use size that makes sense to the image and feature size. Large may not be good. 
#On the synthetic image it is clear how ksize affects imgae (try 5 and 50)
sigma = 3 # Large sigma on small features will fully miss the features. 
theta = 3*np.pi/4  # 1/4 shows horizontal 
lamda = 3*np.pi/4  # 1/4 works good
gamma=10  #Value of 1 defines spherical. Close to 0 has high aspect ratio
phi = 0  #Phase offset

kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_64F)
plt.imshow(kernel)

img = cv2.imread('image_1328257462_product_4250677708.jpg')
#img = cv2.imread('BSE_Image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)

kernel_resized = cv2.resize(kernel, (400, 400))                    
cv2.imshow('Kernel', kernel_resized)
cv2.imshow('Original Img.', img)
cv2.imshow('Filtered', fimg)
cv2.waitKey(30000) #milliseconds 
cv2.destroyAllWindows()


