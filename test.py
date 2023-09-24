import tensorflow as tf
import cv2
import numpy as np
import os
import csv
from keras.preprocessing.image

saved = tf.keras.models.load_model(r'C:\Users\abdel\Desktop\Samar\Year_4\Term1\Neural Networks and Deep Learning\Labs\Project\model.h5')
TEST_DIR = r'C:\Users\abdel\Desktop\Samar\Year_4\Term1\Neural Networks and Deep Learning\Labs\Project\Test'
IMG_SIZE = 224
testing_data=[]
for img in os.listdir(TEST_DIR):
        path = os.path.join(TEST_DIR, img)
        img_data = cv2.imread(path)
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        test_img = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        test_img = test_img.reshape(IMG_SIZE, IMG_SIZE, 3)
        norm_img = np.zeros((IMG_SIZE,IMG_SIZE))
        img_data = cv2.normalize(test_img,  norm_img, 0, 255, cv2.NORM_MINMAX)
        img_data = np.expand_dims(img_data, axis=0)
        prediction = saved.predict(img_data)[0]
        max_value = max(prediction)
        index = np.where(prediction == max_value)
        testing_data.append([img, index[0][0]])
        
with open('sport.csv','w+') as file:
 myfile = csv.writer(file)
 myfile.writerow(['image_name', 'label'])
 for i in range(len(testing_data)):
    myfile.writerow(testing_data[i])