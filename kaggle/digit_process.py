# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 21:22:11 2020

@author: Orange
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
dataset = pd.read_csv("train.csv")
X_train = dataset.iloc[:,1:785].values
Y_train = dataset.iloc[:,0].values
count = sns.countplot(Y_train)

X_train = X_train.reshape(-1,28, 28)
plt.figure()
plt.imshow(X_train[0])
plt.colorbar()
plt.grid(True)
plt.show()
X_train = np.expand_dims(X_train, -1)




dataset_test = pd.read_csv("test.csv")
dataset_answer = pd.read_csv("sample_submission.csv")
x_test = dataset_answer.iloc[:,0].values 
df_test = dataset_test.values
df_test = df_test.reshape(-1,28, 28)
df_test = np.expand_dims(df_test, -1)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.optimizers import Adam


model = Sequential()
## Step 1 - Convolution
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same', 
                 input_shape=X_train[0].shape))
##代表 有32個特徵探測器  有5*5舉證為特徵舉證 輸入舉證為(28,28,1)
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same'))
## Step 2 - Pooling 主要目的就是降維
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
# Adding a second convolutional layer
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', 
                 input_shape=X_train[0].shape))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))
# Step 3 - Flattening
model.add(Flatten())
# Step 4 - Full connection
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

# Compiling the CNN
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer=Adam(0.0001), 
              metrics=['accuracy'])

r = model.fit(X_train, Y_train, epochs=10)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=0.2,  # randomly rotate images in the range 2 degrees
        zoom_range = 0.2, # Randomly zoom image 2%
        width_shift_range=0.2,  # randomly shift images horizontally 2%
        height_shift_range=0.2,  # randomly shift images vertically 2%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)
batch_size = 150
epochs = 10
r.model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                    steps_per_epoch=len(X_train) /150, epochs=epochs)


#plt.plot(history.history, color='r')


#Y_pred = model.predict(X_val)
## Convert predictions classes to one hot vectors 
#Y_pred_classes = np.argmax(Y_pred,axis = 1) 
## Convert validation observations to one hot vectors
#Y_true = np.argmax(Y_val,axis = 1) 
## compute the confusion matrix
#from sklearn.metrics import confusion_matrix
#confusion_mtx = confusion_matrix(Y_val, Y_pred_classes) 
## plot the confusion matrix
#f,ax = plt.subplots(figsize=(8, 8))
#sns.heatmap(confusion_mtx, annot=True, cmap="Greens",linecolor="gray", fmt= '.1f')
#plt.xlabel("Predicted Label")
#plt.ylabel("True Label")
#plt.title("Confusion Matrix")
#plt.show()
r.model.save("Digit.h5")
y_pred = r.model.predict(df_test)
y_pred = np.argmax(y_pred, axis=1)
result = pd.DataFrame(data = {'ImageId':x_test, 'Label':y_pred})
result.to_csv("Submission.csv", index=False)