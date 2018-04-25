import os
import numpy as np
import tensorflow as tf
import random as rn

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Reshape
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import np_utils, to_categorical

#make results reproducible
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1234)
tf.set_random_seed(1234)
rn.seed(1234)

#paths to datasets
DATASET_PATH_POS = "dna_5000_100_1_['TAAAGCGTAATA'].fa"
DATASET_PATH_NEG = "dna_5000_100_0_[].fa"

#one-hot embed the bases
onehotEmbedded = {
'a' : [1,0,0,0],
'c' : [0,1,0,0],
'g' : [0,0,1,0],
't' : [0,0,0,1]
}

#parse string of acgt to array of numbers.
def parseLine(line):
    dnastring = []
    for char in line:
        dnastring.append(onehotEmbedded[char.lower()])
    return np.array(dnastring).flatten()

#load dataset. Takes path to file, returns np array of categorical values.
def loadDataset(path):
    dataset = []
    fil = open(path)
    for line in fil:
        if not line.startswith(">"):
            dataset.append(parseLine(line.strip()))
    return np.array(dataset)

#load data with motifs
x_trainpos = loadDataset(DATASET_PATH_POS)
y_trainpos = np.ones(5000) #and correct score (1)

#load data with without motif
x_trainneg = loadDataset(DATASET_PATH_NEG)
y_trainneg = np.zeros(5000) #and incorrect score (0)

#merge datasets
x_train = np.concatenate((x_trainpos,x_trainneg),axis=0)
y_train = np.concatenate((y_trainpos,y_trainneg),axis=0)

#make label categorical to avoid bias
y_train = to_categorical(y_train)

''' model.fit already shuffles data.
#shuffle data with same seed (so data and label still match)
np.random.seed(1234)
np.random.shuffle(x_train)
np.random.seed(1234)
np.random.shuffle(y_train)
'''
#split up data so we have 5% of it to test on.
x_test = x_train[8500:]
y_test = y_train[8500:]
x_train = x_train[:8500]
y_train = y_train[:8500]

#Reshape((400,1)) --> (100, 4, 1)

#reshape for convolutional layers
#training and validating. 70/15/15
#maxpooling layer
#dropouts to avoid generalizing
#early stopping to save best epoch


#model data.
model = Sequential()
model.add(Reshape((100,4),input_shape=(400,)))
model.add(Conv1D(1,12))
model.add(MaxPooling1D(pool_size=1))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(50, activation='relu'))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

#compile model.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=25,validation_split=0.176) # 15/85 is roughly 0.176

#evaluate it.
score = model.evaluate(x_test, y_test, batch_size=25)
print(score[0],score[1])