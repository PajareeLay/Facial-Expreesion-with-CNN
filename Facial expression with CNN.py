#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization


#prepare database
path = "D:\Chula\Project\Lab\kaggle\input\challenges-in-representation-learning-facial-expression-recognition-challenge\icml_face_data.csv"
data = pd.read_csv(path)
emotion = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# In[2]:


# Output: Image in right shaped and normalized + labels
def parse_data(data):
    image_array = np.zeros(shape=(len(data), 48, 48, 1))
    image_label = np.array(list(map(int, data['emotion'])))
    
    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48, 1))
        image_array[i] = image
        
    return image_array, image_label

# Splitting the data into train, validation and testing set 
X_train, y_train = parse_data(data[data[" Usage"] == "Training"])
X_val, y_val = parse_data(data[data[" Usage"] == "PrivateTest"])
X_test, y_test = parse_data(data[data[" Usage"] == "PublicTest"])


# In[3]:


#Augmentation

from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
rotation_range=0.05,  #Randomly rotate images in the range
zoom_range = 0.2, # Randomly zoom image
width_shift_range=0.1,  #Randomly shift images horizontally
height_shift_range=0.1,  #Randomly shift images vertically
shear_range=0.05 #Randomly shear images
)
datagen.fit(X_train)


# In[ ]:


# Building a CNN model
# Conv#1> Pool#1 > Conv#2 > Pool#2 > Conv#3 > Pool#3 > FC

model = keras.Sequential()

model.add(layers.Conv2D(input_shape=(48,48,1) , 
                        filters=16, kernel_size=(3, 3), activation='relu', name = 'conv1'))
model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu',name = 'conv2'))
model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu',name = 'conv3'))
model.add(layers.MaxPooling2D())


model.add(layers.Flatten())
model.add(layers.Dense(units=120, activation='relu'))
model.add(layers.Dense(units=84, activation='relu'))
model.add(layers.Dense(units=7, activation = 'softmax'))

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer=keras.optimizers.Adam(lr=1e-3), metrics=['accuracy'])

model.summary()


# In[ ]:


# Training the model, and validatin g
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
shuffle=True,
epochs=30, validation_data = (X_val, y_val),
verbose = 1, steps_per_epoch=X_train.shape[0] // 32)


# In[ ]:


#plot accuracy 
plt.figure(figsize=(9,6))
plt.plot(model.history.history['accuracy'], label='Train accuracy')
plt.plot(model.history.history['val_accuracy'], label='Validation accuracy')
plt.ylabel('Value')
plt.xlabel('No. epoch')
plt.legend()
plt.show()


# In[12]:


model.save('model.h5')


# In[ ]:




