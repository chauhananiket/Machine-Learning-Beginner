from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical

from keras import backend as K

from PIL import Image # used for loading images
import numpy as np
import os # used for navigating to image path
import imageio # used for writing images
import cv2

def size_stat(DIR):
  heights = []
  widths = []
  ct = 0
  for catagory in os.listdir(DIR):
    folder = os.path.join(DIR, catagory)
    for img in os.listdir(folder):
        #print(img)
        path = os.path.join(folder, img)
        data = np.array(Image.open(path)) #PIL Image library
        heights.append(data.shape[0])
        widths.append(data.shape[1])
        ct += 1
  avg_height = sum(heights) / len(heights)
  avg_width = sum(widths) / len(widths)
  print("Total images: " + str(ct) + '\n')
  print("Average Height: " + str(avg_height))
  print("Max Height: " + str(max(heights)))
  print("Min Height: " + str(min(heights)))
  print('\n')
  print("Average Width: " + str(avg_width))
  print("Max Width: " + str(max(widths)))
  print("Min Width: " + str(min(widths)))
  

size_stat(r'F:\Python Programs - ML\Dataset\Unzipped\Tamil_Character_Image_Classifier\trainingData') 

img_width, img_height = 27, 27
  
train_data_dir = r'F:\Python Programs - ML\Dataset\Unzipped\Tamil_Character_Image_Classifier\trainingData'
validation_data_dir = r'F:\Python Programs - ML\Dataset\Unzipped\Tamil_Character_Image_Classifier\validationData'
nb_train_samples = 450 
nb_validation_samples = 50
epochs = 30
batch_size = 10

if K.image_data_format() == 'channels_first':  #means rgb is present as a 1st size  
    input_shape = (3, img_width, img_height) 
else: 
    input_shape = (img_width, img_height, 3) 
    
train_datagen = ImageDataGenerator( rescale = 1. / 255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True) 

test_datagen = ImageDataGenerator(rescale = 1. / 255)

train_generator = train_datagen.flow_from_directory(train_data_dir, 
                                                    target_size =(img_width, img_height), 
                                                    batch_size = batch_size, 
                                                    class_mode ='categorical') 
print(len(train_generator.filenames))
print(train_generator.class_indices)
print(len(train_generator.class_indices))
num_classes = len(train_generator.class_indices)   

validation_generator = test_datagen.flow_from_directory(validation_data_dir, 
                                                        target_size =(img_width, img_height), 
                                                        batch_size = batch_size, class_mode ='categorical') 


model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape = input_shape)) 
model.add(Activation('relu')) 
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss ='categorical_crossentropy', optimizer ='rmsprop', metrics =['accuracy'])
model.summary()

model.fit_generator(train_generator, 
                    steps_per_epoch = nb_train_samples , #batch_size, 
                    epochs = epochs, 
                    validation_data = validation_generator, 
                    validation_steps = nb_validation_samples)  # batch_size

image_path = r'F:\Python Programs - ML\Dataset\Unzipped\Tamil_Character_Image_Classifier\Ta (91).jpg'

orig = cv2.imread(image_path)
print("[INFO] loading and preprocessing image...")
image = load_img(image_path, target_size=(27, 27))
image = img_to_array(image)
# important! otherwise the predictions will be '0'
image = image / 255
image = np.expand_dims(image, axis=0)

#prediction = model.predict_classes(image)
#print(prediction)
class_predicted = model.predict_classes(image)

probabilities = model.predict_proba(image)

print(probabilities)

inID = class_predicted[0]

inv_map = {v: k for k, v in train_generator.class_indices.items()}

label = inv_map[inID]

# get the prediction label
print("Image ID: {}, Label: {}".format(inID, label))