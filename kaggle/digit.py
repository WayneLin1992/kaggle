# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:11:37 2020

@author: Orange
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
dataset = pd.read_csv("train.csv")
x = dataset.iloc[:,1:785].values
y = dataset.iloc[:,0].values

### arrary to the martix -1 mean all to (28,28)
x = x.reshape(-1,28, 28)
plt.figure()
plt.imshow(x[0])
plt.colorbar()
plt.grid(False)
plt.show()

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#x = sc.fit_transform(x)
x = np.expand_dims(x, -1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D
#from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow

## Initialising the CNN
#classifier = Sequential()
#model = tensorflow.keras.models.Sequential([
# tensorflow.keras.layers.Flatten(input_shape=(28, 28)),
#  tensorflow.keras.layers.Dense(128, activation='relu'),
#  tensorflow.keras.layers.Dropout(0.2),
#  tensorflow.keras.layers.Dense(10, activation='softmax')
#])
#model.compile(optimizer='adam',
#              loss='sparse_categorical_crossentropy',
#              metrics=['accuracy'])
#
#r = model.fit(x, y, epochs=10)
##
## Initialising the CNN
#classifier = Sequential()
#
## Step 1 - Convolution
#classifier.add(Convolution2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same', input_shape=x[0].shape))
#
## Step 2 - Pooling
#classifier.add(MaxPooling2D(pool_size = (2, 2)))
#
## Adding a second convolutional layer
#classifier.add(Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', 
#                 input_shape=x[0].shape))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))
#
## Step 3 - Flattening
#classifier.add(Flatten())
#
## Step 4 - Full connection
#classifier.add(Dense(output_dim = 256, activation = 'relu'))
#classifier.add(Dense(output_dim = 10, activation = 'softmax'))
#
## Compiling the CNN
#classifier.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])



model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same', 
                 input_shape=x[0].shape))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', 
                 input_shape=x[0].shape))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer=Adam(0.0001), 
              metrics=['accuracy'])

r = model.fit(x, y, epochs=10)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
X_train2, X_val2, Y_train2, Y_val2 = train_test_split(x, y, test_size = 0.1)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=0.5,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.5, # Randomly zoom image 5%
        width_shift_range=0.5,  # randomly shift images horizontally 5%
        height_shift_range=0.5,  # randomly shift images vertically 5%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x)
#training_set = datagen.flow_from_directory('dataset/training_set',
#                                                 target_size = (28, 28),
#                                                 batch_size = 32,
#                                                 class_mode = 'binary')
model.fit_generator(datagen.flow(X_train2,Y_train2,batch_size = 32),validation_data = (X_val2,Y_val2),
                        steps_per_epoch=x.shape[0],epochs=1,verbose=0)



dataset_test = pd.read_csv("test.csv")
dataset_answer = pd.read_csv("sample_submission.csv")
x_test = dataset_answer.iloc[:,0].values 
df_test = dataset_test.values
df_test = df_test.reshape(-1,28, 28)
df_test = np.expand_dims(df_test, -1)
#plt.figure()
#plt.imshow(df_test[15])
#plt.colorbar()
#plt.grid(False)
#plt.show()


y_pred = model.predict(df_test)
y_pred = np.argmax(y_pred, axis=1)
y_pred 
diff = []
#dif = pd.DataFrame(columns=['i','result_before_y','y_pred'])
result_before = pd.read_csv("Submission.csv")
result_before_y =result_before['Label']
for i in range(2800):
    if result_before_y[i] == y_pred[i]:
        continue
    else:
#        dif.append({'i': i,'result_before_y':result_before_y[i],'y_pred':y_pred[i]},ignore_index=True)
        diff.append(i)

dataset_test = pd.read_csv("test.csv")
dataset_answer = pd.read_csv("sample_submission.csv")
x_test = dataset_answer.iloc[:,0].values 
df_test = dataset_test.values
df_test = df_test.reshape(-1,28, 28)
plt.figure()
plt.imshow(df_test[28])
plt.colorbar()
plt.grid(False)
plt.show()
result = pd.DataFrame(data = {'ImageId':x_test, 'Label':y_pred})
result.to_csv("Submission1.csv", index=False)


