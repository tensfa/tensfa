import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations as comb
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten

dataset = pd.read_csv('Partial_quantarize.csv') #My dataset
print(dataset.columns.values)

pick = np.random.rand(len(dataset)) < 0.7
train = dataset[pick]
test = dataset[~pick]

#ingredient for training/testing the algorithm
coord = ['ra','dec']
cmodel_mags = ['Mag_u','Mag_g','Mag_r','Mag_i','Mag_z']
rad = ['rad_u', 'rad_g', 'rad_r', 'rad_i', 'rad_z']
dered = ['ext_u','ext_g','ext_r','ext_i','ext_z']
dered_color_indices = ['ext_ug','ext_gr','ext_ri','ext_iz']
coindex = ['coindex_u','coindex_g','coindex_r','coindex_i','coindex_z']
cmodel_color_indices = ['ug','gr','ri','iz']
prad50 = ['petroR50_u','petroR50_g','petroR50_r','petroR50_i','petroR50_z']
prad90 = ['petroR90_u','petroR90_g','petroR90_r','petroR90_i','petroR90_z']
#rad = ['petroRad_u','petroRad_g','petroRad_r','petroRad_i','petroRad_z']
#petro_color_indices = ['p_ug','p_gr','p_ri','p_iz']

#training models
model1 = cmodel_mags + cmodel_color_indices
model2 = cmodel_mags + cmodel_color_indices + rad
model3 = cmodel_mags + cmodel_color_indices + rad + coindex
model4 = dered + dered_color_indices
model5 = dered + dered_color_indices + rad
model6 = dered + dered_color_indices + rad + coindex
model7 = cmodel_mags + cmodel_color_indices + dered + dered_color_indices + rad + coindex
fullparms = coord + cmodel_mags + cmodel_color_indices + dered + dered_color_indices + rad + prad50 + prad90 + coindex

print(train[model4].shape,test[model4].shape) #this gives me (70061,9) (29939,9)

def nn_mlp(test, train, labels, k=7):
    ylabel = train['redshift']
    prediction = []
    batch=1
    no_bins = k*100 if k*100 < 1000 else 1000
    max_z = np.max(train['redshift'].values)
    min_z = np.min(train['redshift'].values)
    model = Sequential()
    model.add(Dense(len(labels), input_dim=len(labels), kernel_initializer='normal', use_bias=True, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', use_bias=True))
    model.compile(loss='mean_squared_error', optimizer='adam')
    edges = np.histogram(train['redshift'].values[::batch], bins=no_bins, range=(min_z,max_z))[1]
    edges_with_overflow = np.histogram(train['redshift'].values[::batch], bins=no_bins+1, range=(min_z, max_z))[1]
    model.fit(train[labels].values[::batch], edges_with_overflow[np.digitize(train['redshift'].values[::batch], edges)], epochs=1)
    for point in test[labels].values:
        prediction.append(model.predict([point])[0])
    return np.array(prediction)

pred_4 = nn_mlp(test, train, model4)