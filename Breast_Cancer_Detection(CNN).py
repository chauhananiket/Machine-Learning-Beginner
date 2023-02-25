#Importing Libraries

import pandas as pd

import numpy as np

import os

from glob import glob

import itertools

import fnmatch

import random

import matplotlib.pylab as plt

import seaborn as sns

import cv2

from scipy.misc import imresize, imread

import sklearn

from sklearn import model_selection

from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, learning_curve, GridSearchCV

from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import keras

from keras import backend as K

from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, model_from_json

from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta

from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D

import os

import glob2

from matplotlib import style

style.use("ggplot")



print(os.listdir(r'F:\Python Programs(Machine Learning)\Dataset\Unzipped\Breast_Cancer_Dataset'))


from os import listdir

#print(listdir(r'/home2/aparaye/aniket/keras_tut/Breast_Cancer/IDC_regular_ps50_idx5/')[:10])


#print(listdir(r'/home2/aparaye/aniket/keras_tut/Breast_Cancer/IDC_regular_ps50_idx5/10285'))



imagePatches = glob2.glob(r'F:\Python Programs(Machine Learning)\Dataset\Unzipped\Breast_Cancer_Dataset\IDC_regular_ps50_idx5\**\*.png', recursive = True)

for filename in imagePatches[0:10]:

    print(filename)


#Plot the image

image = cv2.imread(r'F:\Python Programs(Machine Learning)\Dataset\Unzipped\Breast_Cancer_Dataset\IDC_regular_ps50_idx5\10285\1\10285_idx5_x1151_y901_class1.png')

plt.figure(figsize=(16,16))

plt.imshow(image)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))



# Plot multiple images

bunchOfImages = imagePatches

i_ = 0

plt.rcParams['figure.figsize'] = (10.0, 10.0)

plt.subplots_adjust(wspace=0, hspace=0)

for l in bunchOfImages[:25]:

    im = cv2.imread(l)

    im = cv2.resize(im, (50, 50), interpolation = cv2.INTER_AREA) 

    plt.subplot(5, 5, i_+1) #.set_title(l)

    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')

    i_ += 1


def randomImages(a):

    r = random.sample(a, 4)

    plt.figure(figsize=(16,16))

    plt.subplot(131)

    plt.imshow(cv2.imread(r[0]))

    plt.subplot(132)

    plt.imshow(cv2.imread(r[1]))

    plt.subplot(133)

    plt.imshow(cv2.imread(r[2])); 


randomImages(imagePatches)


patternZero =r'*class0.png'

patternOne = r'*class1.png'

classZero = fnmatch.filter(imagePatches, patternZero)

classOne  = fnmatch.filter(imagePatches, patternOne)

print("IDC(-)",classZero[0:5])

print("IDC(+)",classOne[0:5])


def proc_images(lowerIndex,upperIndex):
    """
    Returns two arrays: 
        x is an array of resized images
        y is an array of labels
    """ 
    x = []
    y = []
    WIDTH = 50
    HEIGHT = 50
    for img in imagePatches[lowerIndex:upperIndex]:
        full_size_image = cv2.imread(img)
        x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        if img in classZero:
            y.append(0)
        elif img in classOne:
            y.append(1)
        else:
            return
    return x,y


X,Y = proc_images(0,90)

X1 = np.array(X)


print(X1.shape)


df = pd.DataFrame()

df["images"]=X

df["labels"]=Y

X2=df["images"]

Y2=df["labels"]

print(type(X2))


X2=np.array(X2)

print(X2.shape)


imgs0=[]

imgs1=[]

imgs0 = X2[Y2==0] # (0 = no IDC, 1 = IDC)

imgs1 = X2[Y2==1]

 

def describeData(a,b):

    print('Total number of images: {}'.format(len(a)))

    print('Number of IDC(-) Images: {}'.format(np.sum(b==0)))

    print('Number of IDC(+) Images: {}'.format(np.sum(b==1)))

    print('Percentage of positive images: {:.2f}%'.format(100*np.mean(b)))

    print('Image shape (Width, Height, Channels): {}'.format(a[0].shape))


describeData(X2,Y2)


X=np.array(X)

X=X/255.0


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


import gc

gc.collect()


print(X_train.shape)


print(X_test.shape)


dist = df['labels'].value_counts()

print(dist)


sns.countplot(df['labels'])


#One Hot encoding

# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

from keras.utils.np_utils import to_categorical

y_trainHot = to_categorical(Y_train, num_classes = 2)

y_testHot = to_categorical(Y_test, num_classes = 2)


# Helper Functions  Learning Curves and Confusion Matrix

class MetricsCheckpoint(Callback):
    """Callback that saves metrics after each epoch"""
    def __init__(self, savepath):
        super(MetricsCheckpoint, self).__init__()
        self.savepath = savepath
        self.history = {}
    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        np.save(self.savepath, self.history)

def plotKerasLearningCurve():
    plt.figure(figsize=(10,5))
    metrics = np.load('logs.npy')[()]
    filt = ['acc'] # try to add 'loss' to see the loss learning curve
    for k in filter(lambda x : np.any([kk in x for kk in filt]), metrics.keys()):
        l = np.array(metrics[k])
        plt.plot(l, c= 'r' if 'val' not in k else 'b', label='val' if 'val' in k else 'train')
        x = np.argmin(l) if 'loss' in k else np.argmax(l)
        y = l[x]
        plt.scatter(x,y, lw=0, alpha=0.25, s=100, c='r' if 'val' not in k else 'b')
        plt.text(x, y, '{} = {:.4f}'.format(x,y), size='15', color= 'r' if 'val' not in k else 'b')   
    plt.legend(loc=4)
    plt.axis([0, None, None, None]);
    plt.grid()
    plt.xlabel('Number of epochs')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize = (5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_learning_curve(history):
    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.savefig('./accuracy_curve.png')
    #plt.clf()
    # summarize history for loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.savefig('./loss_curve.png')


#Model Buliding

batch_size = 128

num_classes = 2

epochs = 8

img_rows,img_cols=50,50

input_shape = (img_rows, img_cols, 3)

e = 2


model = Sequential()

model.add(Conv2D(128, kernel_size=(3, 3),activation='relu' , input_shape=input_shape#,strides=e

))

model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))


#model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dense(128, activation='relu'))


#model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='sigmoid'))

model.summary()


model.compile(loss=keras.losses.binary_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])


datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)

        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=True,  # randomly flip images

        vertical_flip=True)  # randomly flip images


a = X_train

b = y_trainHot

c = X_test

d = y_testHot

epochs = 1

history = model.fit_generator(datagen.flow(a,b, batch_size=32),

                        steps_per_epoch=len(a) / 32, 

                              epochs=epochs,validation_data = [c, d])


y_pred = model.predict(c)
Y_pred_classes = np.argmax(y_pred,axis=1) 
Y_true = np.argmax(d,axis=1)
dict_characters = {0: 'IDC(-)', 1: 'IDC(+)'}
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
plot_confusion_matrix(confusion_mtx, classes = list(dict_characters.values())) 
plt.show()

plotKerasLearningCurve()
plt.show()  

plot_learning_curve(history)
plt.show()