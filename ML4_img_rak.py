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
        for lamda in np.arange(0, np.pi, np.pi / 4):   #Range of wavelengths
            for gamma in (0.05, 0.5):   #Gamma values of 0.05 and 0.5
            
                
                gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
#                print(gabor_label)
                ksize=9
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                kernels.append(kernel)
                #Now filter the image and add values to a new column 
                fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
                filtered_img = fimg.reshape(-1)
                df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
                print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                num += 1  #Increment for gabor column label
                
#OTHER FEATURES                 
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
#For this, we need to import the labeled image
labeled_img = cv2.imread('image_688575878_product_57497717.jpg')
#Remember that you can load an image with partial labels 
#But, drop the rows with unlabeled data

labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
labeled_img1 = labeled_img.reshape(-1)
df['Labels'] = labeled_img1

print(df.head())

#df.to_csv("Gabor.csv")


#########################################################

#Define the dependent variable that needs to be predicted (labels)
Y = df["Labels"].values

#Define the independent variables
X = df.drop(labels = ["Labels"], axis=1) 

#Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=20)


from sklearn.ensemble import RandomForestClassifier
# Instantiate model with n number of decision trees
model = RandomForestClassifier(n_estimators = 100, random_state = 42)

# Train the model on training data
model.fit(X_train, y_train)

# verify number of trees used. If not defined above. 
print('Number of Trees used : ', model.n_estimators)
# TESTING THE MODEL BY PREDICTING ON TEST DATA
prediction_test_train = model.predict(X_train)

#Test prediction on testing data. 
prediction_test = model.predict(X_test)
from sklearn import metrics
#First check the accuracy on training data. This will be higher than test data prediction accuracy.
print ("Accuracy on training data = ", metrics.accuracy_score(y_train, prediction_test_train))
#Check accuracy on test dataset. If this is too low compared to train it indicates overfitting on training data.
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))

#Get numerical feature importances
importances = list(model.feature_importances_)

feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp)
import pickle
#Save the trained model as pickle string to disk for future use
filename = "rakuten_im1"
pickle.dump(model, open(filename, 'wb'))

#To test the model on future datasets
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(X)

segmented = result.reshape((img.shape))

from matplotlib import pyplot as plt
plt.imshow(segmented, cmap ='jet')
plt.imsave('rakuten_estim.jpg', segmented, cmap ='jet')

