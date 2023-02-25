import cv2
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns
import h5py
from keras import backend as K
from keras.callbacks import  EarlyStopping, Callback
from keras.utils import np_utils
from keras import optimizers
from keras.models import Sequential, model_from_json
from keras.layers import  Conv2D, MaxPool2D,Activation,Dropout,Flatten,Dense, BatchNormalization
import pandas as pd
import numpy as np

# defining constant values
img_width = 64
img_height = 64
split_size = 0.2
batch_size = 128
channels = 3

train_data = h5py.File(r'D:\Downloads\happy-house-dataset\train_happy.h5', "r")
X_train = np.array(train_data["train_set_x"][:]) 
#print(X_train[0])
y_train = np.array(train_data["train_set_y"][:]) 
y_train = y_train.reshape((1, y_train.shape[0]))

test_data = h5py.File(r'D:\Downloads\happy-house-dataset\test_happy.h5', "r")
X_test = np.array(test_data["test_set_x"][:])
y_test = np.array(test_data["test_set_y"][:]) 
y_test = y_test.reshape((1, y_test.shape[0]))

print("Shape of Training data :{}".format(X_train.shape))
print("Shape of Test data :{}".format(X_test.shape))

X_train = X_train/255.
X_test = X_test/255.
y_train = y_train.T #one hot line coding
y_test = y_test.T

#Visualizing the data
fig=plt.figure(figsize=(8, 8))
columns = 5
rows = 5
for i in range(1, columns*rows +1):
    img = X_train[i]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()

model = Sequential()

model.add(Conv2D(8,(3,3),input_shape=(img_width,img_height,channels),padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(3,3)))

model.add(Conv2D(16,padding='same',kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(32,padding='same',kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


model.add(Conv2D(32,kernel_size=(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))


#Compile Model
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, batch_size=30, epochs=20)

y_predict = model.predict_classes(X_test)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
 
accuracy = accuracy_score(y_test, y_predict)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_predict, average='binary')
 
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1_score)

