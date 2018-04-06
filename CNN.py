# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 19:48:23 2018

@author: Yashad
"""
# Importing keras dependencies
import numpy as np
import tensorflow as tf

#Importing necessary layers for CNN
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from sklearn.metrics import *
#from terminaltables import AsciiTable
import random
import seaborn as sns
# Loading data by keras
from keras.datasets import mnist
# Load pre-shuffled MNIST data into train and test sets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Length of train
print(X_train.shape)
target_names=['0','1','2','3','4','5','6','7','8','9']
# Plotting sample data
from matplotlib import pyplot as plt
plt.imshow(X_train[0])

# Preprocess input data
X_train = X_train[:10000,:,:]
X_test = X_test[:1000,:,:]
Y_train = Y_train[:10000]
Y_test  = Y_test[:1000]
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0],28, 28,1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Preprocess class labels
Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)

# Define four layer model architecture
def fourlayer_CNN():
    model_4layer = Sequential()
    model_4layer.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=[28,28,1]))
    model_4layer.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model_4layer.add(Conv2D(64, (5, 5), activation='relu'))
    model_4layer.add(MaxPooling2D(pool_size=(2, 2)))
    model_4layer.add(Flatten())
    model_4layer.add(Dense(1000, activation='relu'))
    model_4layer.add(Dense(10, activation='softmax'))
    return model_4layer
#
## Compile four layer model
#model_4layer = fourlayer_CNN()
#model_4layer.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(lr=0.01),metrics=['accuracy'])
#
## 9. Fit model on training data
#model_4layer.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1)
# 
## 10. Evaluate model on test data
#score = model_4layer.evaluate(X_test, Y_test, verbose=0)
#
# Define six layer model architecture
def sixlayer_CNN():
    model_6layer = Sequential()
    model_6layer.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=[28,28,1]))
    model_6layer.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model_6layer.add(Conv2D(64, (5, 5), activation='relu'))
    model_6layer.add(MaxPooling2D(pool_size=(2, 2)))
    model_6layer.add(Conv2D(64, (4, 4), activation='relu'))
    model_6layer.add(MaxPooling2D(pool_size=(1, 1)))
    model_6layer.add(Flatten())
    model_6layer.add(Dense(1000, activation='relu'))
    model_6layer.add(Dense(10, activation='softmax'))
    return model_6layer

# Compile six layer model
model_6layer = sixlayer_CNN()
model_6layer.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(lr=0.01),metrics=['accuracy'])
# Start training. 
history = model_6layer.fit(X_train, Y_train, batch_size=300, nb_epoch=20, 
                    verbose=1, validation_data=(X_test, Y_test))

# Once finished, we can score our model and print the overall accuracy.
score = model_6layer.evaluate(X_test, Y_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# For more details regarding the model's accuracy, we can compute
# a number of other metrics. For this we need the predicted labels
# and true labels for each image in the test set:
y_pred = model_6layer.predict_classes(X_test) # This will take a few seconds...
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