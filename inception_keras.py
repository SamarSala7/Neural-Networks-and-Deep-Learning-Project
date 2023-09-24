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
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR, img)
        img_data = cv2.imread(path)
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        norm_img = np.zeros((IMG_SIZE,IMG_SIZE))
        img_data = cv2.normalize(img_data,  norm_img, 0, 255, cv2.NORM_MINMAX)
        training_data.append([np.array(img_data), create_label(img)])
    
    shuffle(training_data)
    return training_data

def create_test_labeled_data():
    test_labeled_data = []
    for img in tqdm(os.listdir(TEST_LABELED_DIR)):
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        path = os.path.join(TEST_LABELED_DIR, img)
        img_data = cv2.imread(path)
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        norm_img = np.zeros((IMG_SIZE,IMG_SIZE))
        img_data = cv2.normalize(img_data,  norm_img, 0, 255, cv2.NORM_MINMAX)
        test_labeled_data.append([np.array(img_data), create_label(img)])
    
    shuffle(test_labeled_data)
    return test_labeled_data



def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj, name=None):
    
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')(x)
    
    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu')(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu')(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu')(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    
    return output


TRAIN_DIR = '/kaggle/input/nn23-sports-image-classification/Train'
TEST_DIR = '/kaggle/input/nn23-sports-image-classification/Test'
TEST_LABELED_DIR = '/kaggle/input/test-labeled/Test_labeled'

IMG_SIZE = 224
LR = 0.001
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





input_layer = Input(shape=(224, 224, 3))

x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2')(input_layer)
x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=64,
                     filters_3x3_reduce=96,
                     filters_3x3=128,
                     filters_5x5_reduce=16,
                     filters_5x5=32,
                     filters_pool_proj=32,
                     name='inception_3a')

x = inception_module(x,
                     filters_1x1=128,
                     filters_3x3_reduce=128,
                     filters_3x3=192,
                     filters_5x5_reduce=32,
                     filters_5x5=96,
                     filters_pool_proj=64,
                     name='inception_3b')

x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=192,
                     filters_3x3_reduce=96,
                     filters_3x3=208,
                     filters_5x5_reduce=16,
                     filters_5x5=48,
                     filters_pool_proj=64,
                     name='inception_4a')


x1 = AveragePooling2D((5, 5), strides=3)(x)
x1 = Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
x1 = Flatten()(x1)
x1 = Dense(1024, activation='relu')(x1)
x1 = Dropout(0.7)(x1)
x1 = Dense(6, activation='softmax', name='auxilliary_output_1')(x1)

x = inception_module(x,
                     filters_1x1=160,
                     filters_3x3_reduce=112,
                     filters_3x3=224,
                     filters_5x5_reduce=24,
                     filters_5x5=64,
                     filters_pool_proj=64,
                     name='inception_4b')

x = inception_module(x,
                     filters_1x1=128,
                     filters_3x3_reduce=128,
                     filters_3x3=256,
                     filters_5x5_reduce=24,
                     filters_5x5=64,
                     filters_pool_proj=64,
                     name='inception_4c')

x = inception_module(x,
                     filters_1x1=112,
                     filters_3x3_reduce=144,
                     filters_3x3=288,
                     filters_5x5_reduce=32,
                     filters_5x5=64,
                     filters_pool_proj=64,
                     name='inception_4d')


x2 = AveragePooling2D((5, 5), strides=3)(x)
x2 = Conv2D(128, (1, 1), padding='same', activation='relu')(x2)
x2 = Flatten()(x2)
x2 = Dense(1024, activation='relu')(x2)
x2 = Dropout(0.7)(x2)
x2 = Dense(6, activation='softmax', name='auxilliary_output_2')(x2)

x = inception_module(x,
                     filters_1x1=256,
                     filters_3x3_reduce=160,
                     filters_3x3=320,
                     filters_5x5_reduce=32,
                     filters_5x5=128,
                     filters_pool_proj=128,
                     name='inception_4e')

x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=256,
                     filters_3x3_reduce=160,
                     filters_3x3=320,
                     filters_5x5_reduce=32,
                     filters_5x5=128,
                     filters_pool_proj=128,
                     name='inception_5a')

x = inception_module(x,
                     filters_1x1=384,
                     filters_3x3_reduce=192,
                     filters_3x3=384,
                     filters_5x5_reduce=48,
                     filters_5x5=128,
                     filters_pool_proj=128,
                     name='inception_5b')

x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)

x = Dropout(0.4)(x)

x = Dense(6, activation='softmax', name='output')(x)

model = Model(input_layer, [x, x1, x2], name='inception_v1')

check_point = k.callbacks.ModelCheckpoint(filepath="inception.h5", monitor="val_acc", mode="max",
                                              save_best_only=True,)
model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], 
              loss_weights=[1, 0.3, 0.3], optimizer=k.optimizers.RMSprop(lr=2e-5), metrics=['accuracy'])

history = model.fit(X_train, [y_train, y_train, y_train], 
                    validation_data=(X_test, [y_test, y_test, y_test]), 
                    epochs=500, batch_size=256, callbacks=[check_point])





testing_data=[]
for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_data = cv2.imread(path)
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        test_img = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        test_img = test_img.reshape(IMG_SIZE, IMG_SIZE, 3)
        norm_img = np.zeros((IMG_SIZE,IMG_SIZE))
        img_data = cv2.normalize(test_img,  norm_img, 0, 255, cv2.NORM_MINMAX)
        img_data = np.expand_dims(img_data, axis=0)
        prediction = model.predict(img_data)[0][0]
        max_value = max(prediction)
        index = np.where(prediction == max_value)
        testing_data.append([img, index[0][0]])
        
with open('sport.csv','w+') as file:
 myfile = csv.writer(file)
 myfile.writerow(['image_name', 'label'])
 for i in range(len(testing_data)):
    myfile.writerow(testing_data[i])