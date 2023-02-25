# importing required modules 
from zipfile import ZipFile 
  
# specifying the zip file name 
file_name = r"C:\Users\Aniket Chauhan\Desktop\Python Programs - ML\Dataset\Zipped\horse-or-human.zip"
  
# opening the zip file in READ mode 
#with ZipFile(file_name, 'r') as zip: 
    # printing all the contents of the zip file 
    #zip.printdir() 
    # extracting all the files 
  #  print('Extracting all the files now...') 
   # zip.extractall(r'C:\Users\Aniket Chauhan\Desktop\Python Programs - ML\Dataset\Unzipped\horse-or-human') 
    #print('Done!') 
import os

# Directory with our training horse pictures
train_horse_dir = os.path.join(r'C:\Users\Aniket Chauhan\Desktop\Python Programs - ML\Dataset\Unzipped\horse-or-human\horses')

# Directory with our training human pictures
train_human_dir = os.path.join(r'C:\Users\Aniket Chauhan\Desktop\Python Programs - ML\Dataset\Unzipped\horse-or-human\humans')    

train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])

print('total training horse images:', len(os.listdir(train_horse_dir)))
print('total training human images:', len(os.listdir(train_human_dir)))
# Directory with our training horse pictures
validation_horse_dir = os.path.join(r'C:\Users\Aniket Chauhan\Desktop\Python Programs - ML\Dataset\Unzipped\validation-horse-or-human\horses')

# Directory with our training human pictures
validation_human_dir = os.path.join(r'C:\Users\Aniket Chauhan\Desktop\Python Programs - ML\Dataset\Unzipped\validation-horse-or-human\humans')

#matplotlib inline
validation_horse_hames = os.listdir(validation_horse_dir)
print(validation_horse_hames[:10])

validation_human_names = os.listdir(validation_human_dir)
print(validation_human_names[:10])

print('total validation horse images:', len(os.listdir(validation_horse_dir)))
print('total validation human images:', len(os.listdir(validation_human_dir)))

# Parameters for our graph; we'll output images in a 4x4 configuration

import tensorflow as tf
'''We then add convolutional layers as in the previous example, and flatten the final result to feed into the densely connected layers.

Finally we add the densely connected layers.

Note that because we are facing a two-class classification problem, i.e. a binary classification problem, we will end our network with a sigmoid activation, so that the output of our network will be a single scalar between 0 and 1, encoding the probability that the current image is class 1 (as opposed to class 0).

'''
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),# we can reduce image size in input_shape
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])
'''The model.summary() method call prints a summary of the NN'''
model.summary()
'''The "output shape" column shows how the size of your feature map evolves in each successive layer. The convolution layers reduce the size of the feature maps by a bit due to padding, and each pooling layer halves the dimensions.

Next, we'll configure the specifications for model training. We will train our model with the binary_crossentropy loss, because it's a binary classification problem and our final activation is a sigmoid. (For a refresher on loss metrics, see the Machine Learning Crash Course.) We will use the rmsprop optimizer with a learning rate of 0.001. During training, we will want to monitor classification accuracy.

NOTE: In this case, using the RMSprop optimization algorithm is preferable to stochastic gradient descent (SGD), because RMSprop automates learning-rate tuning for us. (Other optimizers, such as Adam and Adagrad, also automatically adapt the learning rate during training, and would work equally well here.)
'''

from tensorflow.keras.optimizers import RMSprop
model.compile(loss='binary_crossentropy',  optimizer=RMSprop(lr=0.001), metrics=['acc'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(
        r'C:\Users\Aniket Chauhan\Desktop\Python Programs - ML\Dataset\Unzipped\horse-or-human',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_datagen = ImageDataGenerator(rescale=1/255)
# Flow training images in batches of 128 using train_datagen generator
validation_generator = validation_datagen.flow_from_directory(
         r'C:\Users\Aniket Chauhan\Desktop\Python Programs - ML\Dataset\Unzipped\validation-horse-or-human',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=8,  
      epochs=3,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8)

import numpy as np
from keras.preprocessing import image
path = r'C:\Users\Aniket Chauhan\Desktop\Python Programs - ML\Dataset\Unzipped\horse-or-human\maxresdefault.jpg'
img = image.load_img(path, target_size=(300, 300))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
classes = model.predict(x, batch_size=10)
print(classes[0])
if classes[0]>0.5:
    print(" is a human")
else:
    print(" is a horse")





