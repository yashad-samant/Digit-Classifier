# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 18:16:18 2018

@author: Yashad
"""

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import *
#from terminaltables import AsciiTable
import random
import seaborn as sns
# Loading data by keras

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
target_names=['0','1','2','3','4','5','6','7','8','9']
# Loading data by keras
from keras.datasets import mnist
# Load pre-shuffled MNIST data into train and test sets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Length of train
print(X_train.shape)

# Plotting sample data
from matplotlib import pyplot as plt
plt.imshow(X_train[0])

# Preprocess input data
X_train = X_train[:10000,:,:]
X_test = X_test[:1000,:,:]
Y_train = Y_train[:10000]
Y_test  = Y_test[:1000]
X_train = X_train.reshape(X_train.shape[0],784)
X_test = X_test.reshape(X_test.shape[0],784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Preprocess class labels
Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)


# define baseline model
def baseline_model():
	# create model
    model = Sequential()
    model.add(Dense(1000, input_dim=784, activation='relu'))
    model.add(Dense(100, input_dim=784, activation='relu'))
    model.add(Dense(10, activation='softmax'))
	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
model = baseline_model()
history = model.fit(X_train, Y_train, batch_size=100, nb_epoch=2, 
                    verbose=1, validation_data=(X_test, Y_test))

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(model, X_train, Y_train, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

y_pred = model.predict_classes(X_test) # This will take a few seconds...
Y_pred = np_utils.to_categorical(y_pred, 10) 


# Compute a confusion matrix and a normalised confusion matrix
cm = confusion_matrix(Y_test.argmax(axis=1), Y_pred.argmax(axis=1))
cm_normalised = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# These can also be used to produce heatmaps using Matplotlib or Seaborn.
print(cm)
print(cm_normalised)

# Print a classification report, including precision, recall, and f1-score.
print(classification_report(Y_test, Y_pred, target_names=['0','1','2','3','4','5','6','7','8','9'])) 

np.savetxt('y_true.txt', Y_test)
np.savetxt('y_pred.txt', y_pred)

# Plot a confusion matrix graphically 
#sns.set(font_scale=4.5) 
fig, ax = plt.subplots(figsize=(30,20))
ax = sns.heatmap(cm, annot=True, linewidths=2.5, square=True, linecolor="Green", 
                    cmap="Greens", yticklabels=target_names, xticklabels=target_names, vmin=0, vmax=900, 
                    fmt="d", annot_kws={"size": 50})
ax.set(xlabel='Predicted label', ylabel='True label')

# Get values for plotting:
acc = history.history['acc']
loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

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