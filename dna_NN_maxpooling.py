import numpy as np
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten, Dense, Activation
from keras.utils import np_utils
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split

f0 = open("dna_5000_100_0_[].fa",'r')
f1 = open("dna_5000_100_1_['TAAAGCGTAATA'].fa",'r')


def create_matrix(file):
    returnval = np.ndarray(500000,)
    x = 0
    for line in file:
        if line.startswith(">"):
            continue
        for s in line:
            s = s.lower()
            if s == "a":
                returnval[x] = 0
            elif s == "c":
                returnval[x] = 1
            elif s == "g":
                returnval[x] = 2
            elif s == "t":
                returnval[x] = 3
            elif s == "\n":
                continue
            else:
                print("unexpexted string value found: " + s)
            x += 1

    return returnval
no_motif_data = create_matrix(f0)
with_motif_data = create_matrix(f1)


no_motif_data = np_utils.to_categorical(list(no_motif_data))
no_motif_data = no_motif_data.reshape(5000,100,4,1)

with_motif_data = np_utils.to_categorical(list(with_motif_data))
with_motif_data = with_motif_data.reshape(5000,100,4,1)

seed = 7
np.random.seed(seed)

no_motif_train, no_motif_test, with_motif_train, with_motif_test = train_test_split(no_motif_data, with_motif_data, test_size=0.33, random_state=seed)



data = np.concatenate((no_motif_train,with_motif_train), axis=0)
pred_data = np.concatenate((no_motif_test,with_motif_test,),axis=0)

print(data.shape)

data_size = data.shape[0]

no_motif_label = np.zeros((int(data_size/2),1))
with_motif_label = np.ones((int(data_size/2),1))

labels = np.concatenate((no_motif_label,with_motif_label), axis=0)

labels = np_utils.to_categorical(list(labels))

print(labels)
#Create model
model = Sequential()

model.add(Convolution2D(64, kernel_size=(12, 4), activation='relu', input_shape=(100,4,1)))
model.add(MaxPooling2D(pool_size=(12,4),activation="relu"))

model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
model.summary()
