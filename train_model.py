import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten, Dense
from keras.layers import MaxPooling2D, Dropout
from keras.utils import np_utils
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split
from keras.callbacks import Callback, ModelCheckpoint


#Choose the data to train on
model_choice = "model1.h5"
if model_choice == "model1.h5":
    f0 = open("dna_5000_100_0_[].fa", 'r')
    f1 = open("dna_5000_100_1_['TAAAGCGTAATA'].fa", 'r')
if model_choice == "model2.h5":
    f0 = open("seq2_dna_5000_100_0_[].fa",'r')
    f1 = open("seq2_dna_5000_100_1_['CTCTTGAGG'].fa",'r')



#formatting data

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

seed = 18
np.random.seed(seed)

no_motif_train, no_motif_test, with_motif_train, with_motif_test = train_test_split(no_motif_data, with_motif_data, test_size=0.15, random_state=seed)



data = np.concatenate((no_motif_train,with_motif_train), axis=0)
test_data = np.concatenate((no_motif_test,with_motif_test,),axis=0)


data_size = data.shape[0]
test_data_size = test_data.shape[0]

#creating labels
no_motif_label = np.zeros((int(data_size/2),1))
with_motif_label = np.ones((int(data_size/2),1))

test_no_motif_label = np.zeros((int(test_data_size/2),1))
test_with_motif_label = np.ones((int(test_data_size/2),1))

labels = np.concatenate((no_motif_label,with_motif_label), axis=0)
test_labels = np.concatenate((test_no_motif_label,test_with_motif_label), axis=0)

#making labels categorical
labels = np_utils.to_categorical(list(labels))
test_labels = np_utils.to_categorical(list(test_labels))


#Create model

#model structured based on recommendations from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4908339/#!po=31.1594

#only change is following the convention of equally many hidden nodes in each layer. thus we use 32 convolutional layers instead of 16 to match
#the 32 dense layer (this model outperforms the one with 16)
#  ^test this for new models
#in addition a window size of 12 is used. is this cheating?


#saving the val_loss for each epoch
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

model = Sequential()
model.add(Convolution2D(64, kernel_size=(12, 4), activation='relu', input_shape=(100,4,1)))
model.add(MaxPooling2D(pool_size=(89,1)))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(32,activation='relu'))
model.add(Flatten())
model.add(Dense(2,activation='softmax'))





model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

#saving the best epoch
checkpointer = ModelCheckpoint(filepath='tmp\weights.hdf5', verbose=1, save_best_only=True)
#printing summary of model
model.summary()

#training our model
model.fit(data, labels,shuffle=True,epochs=10, verbose=2, batch_size=20,callbacks=[checkpointer], validation_split=0.1764) #validation split results in 70/15/15 train/validation/test

#saving our model for later use
model.save(model_choice)