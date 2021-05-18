from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Activation, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import random

df = pd.read_csv(r'UTKFace\data.csv', low_memory=False, header=None)
df = df.values

labels = [];
data = [];
random.seed(42)
random.shuffle(df);

for i in range(0,df.shape[0]):
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread("UTKFace/" + df[i][0])
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    labels.append(df[i][1])

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.3, random_state=42)

trainY = to_categorical(trainY, 2)
testY = to_categorical(testY, 2)