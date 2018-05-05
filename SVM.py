# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:19:17 2018

@author: Yashad
"""

# Loading data by keras
from keras.datasets import mnist
from keras.utils import np_utils
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import *
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
# Load pre-shuffled MNIST data into train and test sets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Length of train
print(X_train.shape)
target_names=['0','1','2','3','4','5','6','7','8','9']
# Plotting sample data
from matplotlib import pyplot as plt
plt.imshow(X_train[0])


X_train=X_train.reshape(60000,784)
X_test=X_test.reshape(10000,784)
# Preprocess input data
X_val = X_test[1001:2000]
Y_val = Y_test[1001:2000]
X_train = X_train[:10000]
X_test = X_test[:1000]
Y_train = Y_train[:10000]
Y_test  = Y_test[:1000]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_val = X_val.astype('float32')
X_train /= 255
X_test /= 255
X_val /= 255


# Preprocess class labels
#Y_train = np_utils.to_categorical(Y_train, 10)
#Y_test = np_utils.to_categorical(Y_test, 10)

#Sklearn SVM model building
#Linear model
estimator_linear = svm.SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.0001, kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=True)
estimator_linear.fit(X_train,Y_train)


params_dict = {"C": np.logspace(-1, 100, 5), "gamma": np.linspace(0.0001, 10, 5)}


 #Fit the grid search
#search = GridSearchCV(estimator=estimator_linear, param_grid=params_dict)
#search.fit(X_val, Y_val)
#print("Best parameter values:", search.best_params_)
#print("CV Score with best parameter values:", search.best_score_)
#df = pd.DataFrame(search.cv_results_)
#df.head()
Y_pred = estimator_linear.predict(X_test)
score = estimator_linear.score(X_test, Y_test)
cm = confusion_matrix(Y_test, Y_pred)
cm_normalised = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# These can also be used to produce heatmaps using Matplotlib or Seaborn.
print(cm)
print(cm_normalised)

# Print a classification report, including precision, recall, and f1-score.
print(classification_report(Y_test, Y_pred, target_names=['0','1','2','3','4','5','6','7','8','9'])) 

np.savetxt('y_true.txt', Y_test)
np.savetxt('y_pred.txt', Y_pred)

# Plot a confusion matrix graphically 
#sns.set(font_scale=4.5) 
fig, ax = plt.subplots(figsize=(30,20))
ax = sns.heatmap(cm, annot=True, linewidths=2.5, square=True, linecolor="Green", 
                    cmap="Greens", yticklabels=target_names, xticklabels=target_names, vmin=0, vmax=900, 
                    fmt="d", annot_kws={"size": 50})
ax.set(xlabel='Predicted label', ylabel='True label')

# Get values for plotting:
acc = accuracy_score(Y_test, Y_pred)
loss = mean_squared_error(Y_test, Y_pred)


# Plots
fig, ax1 = plt.subplots()
plt.grid(True)
ax1.plot(acc, 'g-', linewidth=2.0, label="Accuracy")
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy', color='g')
for tl in ax1.get_yticklabels():
    tl.set_color('g')
ax2 = ax1.twinx()
# Here we plot a point at 0,0, give it the label Accuracy and add the legend... it's a hack to get 2 labels
ax2.plot(0, 0, 'g-', label="Accuracy", linewidth=2.0)
ax2.plot(loss, 'r-', linewidth=2.0, label="Loss")
ax2.set_ylabel('Loss', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
plt.xlim((1,20))
ax2.legend(loc='center right')
plt.show()