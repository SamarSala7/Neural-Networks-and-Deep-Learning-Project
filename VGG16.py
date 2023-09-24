import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import os
from random import shuffle
from tqdm import tqdm
import tensorflow.keras as k
import tensorflow as tf
import csv
from keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Input, concatenate, GlobalAveragePooling2D, AveragePooling2D, Flatten

def create_label(image_name):
    """ Create an one-hot encoded vector from image name """
    word_label = image_name.split('_')
    if word_label[0] == 'Basketball':
        return np.array([1,0,0,0,0,0])
    elif word_label[0] == 'Football':
        return np.array([0,1,0,0,0,0])
    elif word_label[0] == 'Rowing':
        return np.array([0,0,1,0,0,0])
    elif word_label[0] == 'Swimming':
        return np.array([0,0,0,1,0,0])
    elif word_label[0] == 'Tennis':
        return np.array([0,0,0,0,1,0])
    elif word_label[0] == 'Yoga':
        return np.array([0,0,0,0,0,1])

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR, img)
        img_data = cv2.imread(path)
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        norm_img = np.zeros((IMG_SIZE,IMG_SIZE))
        img_data = cv2.normalize(img_data,  norm_img, 0, 255, cv2.NORM_MINMAX)
        img_data = img_data.astype('float')
        training_data.append([np.array(img_data), create_label(img)])
    
    shuffle(training_data)
    return training_data

def create_test_labeled_data():
    test_labeled_data = []
    for img in tqdm(os.listdir(TEST_LABELED_DIR)):
        path = os.path.join(TEST_LABELED_DIR, img)
        img_data = cv2.imread(path)
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        norm_img = np.zeros((IMG_SIZE,IMG_SIZE))
        img_data = cv2.normalize(img_data,  norm_img, 0, 255, cv2.NORM_MINMAX)
        img_data = img_data.astype('float')
        test_labeled_data.append([np.array(img_data), create_label(img)])
    
    shuffle(test_labeled_data)
    return test_labeled_data


TRAIN_DIR = '/kaggle/input/nn23-sports-image-classification/Train'
TEST_DIR = '/kaggle/input/nn23-sports-image-classification/Test'
TEST_LABELED_DIR = '/kaggle/input/test-labeled/Test_labeled'

IMG_SIZE = 100
MODEL_NAME = 'Sports-cnn'

train_data = create_train_data()
test_labeled_data = create_test_labeled_data()


train = train_data
test_labeled = test_labeled_data

X_train  = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_train  = np.array([i[1] for i in train])

X_test = np.array([i[0] for i in test_labeled_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_test = np.array([i[1] for i in test_labeled])

k.utils.to_categorical(y_train,6)
k.utils.to_categorical(y_test,6)

print('preprocessed :)')




model = k.models.Sequential()
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=6, activation="softmax"))

check_point = k.callbacks.ModelCheckpoint(filepath="VGG16.h5", monitor="val_acc", mode="max",
                                              save_best_only=True,)

model.compile(optimizer=k.optimizers.Adam(lr=2e-5), 
              loss=k.losses.categorical_crossentropy, 
              metrics=['accuracy'])
history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test), 
                    epochs=500, batch_size=256, callbacks=[check_point])