# Importing the libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

# Load the dataset
trainp = r'F:\Python Programs(Machine Learning)\Dataset\Unzipped\Dog_Cat_Classifier\train'
testp = r'F:\Python Programs(Machine Learning)\Dataset\Unzipped\Dog_Cat_Classifier\test'

imgo = ImageDataGenerator(rescale = 1./255)
Xy_train = imgo.flow_from_directory(trainp,target_size = (64, 64),
                                    batch_size =64,class_mode = 'binary')
Xy_test = imgo.flow_from_directory(testp,target_size = (64, 64),
                                   batch_size = 32,class_mode = 'binary')

# Define the model
model = tf.keras.Sequential()

model.add(Conv2D(64, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

# Compile the model
opt = tf.train.AdamOptimizer()
model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

# Train the model
hist = model.fit_generator(Xy_train,epochs =5,validation_data =Xy_test)

# Plot the curves
acc=hist.history['acc']
val_acc=hist.history['val_acc']
loss=hist.history['loss']
val_loss=hist.history['val_loss']

n = len(acc)

plt.plot(n, acc, 'r', "Training Accuracy")
plt.plot(n, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.show()

plt.plot(n, loss, 'r', "Training Loss")
plt.plot(n, val_loss, 'b', "Validation Loss")
plt.title('Training and validation accuracy')
plt.show()    