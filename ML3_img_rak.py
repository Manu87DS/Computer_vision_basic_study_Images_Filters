# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 11:43:28 2021
@author: Emmanuel Bonnet
"""
import numpy as np
import cv2
import pandas as pd
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage as nd

img = cv2.imread('image_688575878_product_57497717.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
#### Multiple images can be used for training. For that, you need to concatenate the data
img2 = img.reshape(-1)
df = pd.DataFrame()
df['Original Image'] = img2
#Generate Gabor features
num = 1  #To count numbers up in order to give Gabor features a label in dataframe
kernels = []
for theta in range(2):   #Define number of thetas
    theta = theta / 4. * np.pi
    for sigma in (1, 3):  #Sigma with 1 and 3
        for lamda in np.arange(0, np.pi, np.pi / 4):  
            for gamma in (0.05, 0.5):   
                gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2...
                ksize=9
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                kernels.append(kernel)
                #Now filter the image and add values to a new column 
                fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
                filtered_img = fimg.reshape(-1)
                df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
                print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                num += 1  #Increment for gabor column label
                
#COMPLEMENTARY FILTERS                 
#CANNY EDGE
edges = cv2.Canny(img, 100,200).reshape(-1)  
df['Canny'] = edges 
#ROBERTS EDGE
edge_roberts = roberts(img).reshape(-1)
df['Roberts'] = edge_roberts
#SOBEL
edge_sobel = sobel(img).reshape(-1)
df['Sobel'] = edge_sobel
#SCHARR
edge_scharr = scharr(img).reshape(-1)
df['Scharr'] = edge_scharr
#PREWITT
edge_prewitt = prewitt(img).reshape(-1)
df['Prewitt'] = edge_prewitt
#GAUSSIAN with sigma=3
gaussian_img = nd.gaussian_filter(img, sigma=3).reshape(-1)
df['Gaussian_sig3'] = gaussian_img
#GAUSSIAN with sigma=6
gaussian_img2 = nd.gaussian_filter(img, sigma=6).reshape(-1)
df['Gaussian_sig6'] = gaussian_img2
#MEDIAN with sigma=3
median_img = nd.median_filter(img, size=3).reshape(-1)
df['Median s3'] = median_img
             
#Now, add a column in the data frame for the Labels
labeled_img = cv2.imread('image_688575878_product_57497717.jpg')
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY).reshape(-1)
df['Labels'] = labeled_img
print(df.head())
df.to_csv("Gabor3.csv")

