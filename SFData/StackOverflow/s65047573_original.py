import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.utils import shuffle

import cv2
import warnings

warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv1D, MaxPool1D

from tensorflow.keras.layers import Dense, Dropout, Activation, Input, BatchNormalization, GlobalAveragePooling2D

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
training_folder = r"F:\Pycharm_projects\Kaggle Cassava\data\train_images"
samples_df = pd.read_csv(r"F:\Pycharm_projects\Kaggle Cassava\data\train.csv")
samples_df = shuffle(samples_df, random_state=42)
samples_df["label"] = samples_df["label"].astype("str")
samples_df.head()
temp_labels = {}
imgg = []
lab = []
for i in range(len(samples_df)):
    image_name = samples_df.iloc[i, 0]
    image_label = samples_df.iloc[i, 1]
    la = {image_name: image_label}
    temp_labels.update(la)
print(len(temp_labels))
for im in tqdm(os.listdir(training_folder)):
    path = os.path.join(training_folder, im)
    label = temp_labels.get(im)
    img = cv2.imread(path)
    img = tf.image.random_crop(img, size=(150, 150, 3))
    imgg.append(img)
    lab.append(label)

lables = np.array(lab).astype(np.float32)
img = np.array(imgg).astype(np.float32)
train = tf.data.Dataset.from_tensor_slices((img, lables)).shuffle(buffer_size=1000)
print(tf.data.Dataset.cardinality(train))
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=2, strides=1, activation="relu"))
model.add(BatchNormalization())

model.add(Conv1D(filters=16, kernel_size=2, strides=1, activation="relu"))
model.add(BatchNormalization())

model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(5, activation="sigmoid"))

tf.keras.optimizers.Adam(
    learning_rate=0.0001, )
model.compile(optimizer='adam',
              loss="categorical_crossentropy"
              ,
              metrics=['accuracy'])
model.fit(train, batch_size=32, shuffle=True, epochs=1)