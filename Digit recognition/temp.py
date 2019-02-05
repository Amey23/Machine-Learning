# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_csv("C:/Users/ASUS/Desktop/data/train.csv")

#print(dataset)

clf = DecisionTreeClassifier()

#Training Datasets

xtrain = dataset.iloc[0:21000,1:].values
train_label = dataset.iloc[0:21000,0].values

clf.fit(xtrain, train_label)

#Testing Data
xtest = dataset.iloc[21000:,1:].values
actual_label = dataset.iloc[21000:,0].values

#sample data
d = xtest[90] #can use any index below 42000
print(d)
d.shape = (28,28)
plt.imshow(255-d,cmap = "gray") #we have 255-d because I want white background with black colour
plt.show()
print(clf.predict([xtest[90]]))

#accuracy
p = clf.predict(xtest) #can't pass d because it only takes single row vector
count = 0
for i in range(0,21000):
 if(p[i]==actual_label[i]):
   count += 1 
print("ACCURACY", (count/21000) * 100)
   
   
   