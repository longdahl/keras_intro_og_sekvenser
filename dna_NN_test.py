import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten, Dense
from keras.layers import MaxPooling2D, Dropout, Reshape, GlobalMaxPool2D
from keras.utils import np_utils
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split
import h5py

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

seed = 21
np.random.seed(seed)

no_motif_train, no_motif_test, with_motif_train, with_motif_test = train_test_split(no_motif_data, with_motif_data, test_size=0.15, random_state=seed)



data = np.concatenate((no_motif_train,with_motif_train), axis=0)
test_data = np.concatenate((no_motif_test,with_motif_test,),axis=0)


data_size = data.shape[0]
test_data_size = test_data.shape[0]

no_motif_label = np.zeros((int(data_size/2),1))
with_motif_label = np.ones((int(data_size/2),1))

test_no_motif_label = np.zeros((int(test_data_size/2),1))
test_with_motif_label = np.ones((int(test_data_size/2),1))

labels = np.concatenate((no_motif_label,with_motif_label), axis=0)
test_labels = np.concatenate((test_no_motif_label,test_with_motif_label), axis=0)

labels = np_utils.to_categorical(list(labels))
test_labels = np_utils.to_categorical(list(test_labels))


#Create model

#model structured based on recommendations from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4908339/#!po=31.1594

#only change is following the convention of equally many hidden nodes in each layer. thus we use 32 convolutional layers instead of 16 to match
#the 32 dense layer (this model outperforms the one with 16)
#  ^test this for new models
#in addition a window size of 12 is used. is this cheating?
model = Sequential()
model.add(Convolution2D(32, kernel_size=(12, 4), activation='relu', input_shape=(100,4,1)))
model.add(Reshape((89,32,1),input_shape=(89,1,32)))
model.add(MaxPooling2D(pool_size=(89,1)))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.15))
model.add(Flatten())
model.add(Dense(2,activation='softmax')) #why does it work to have the data as a one_hot vector for binary?



model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
model.summary()

model.fit(data, labels,shuffle=True,epochs=10, verbose=1, batch_size=20,validation_split=0.1764) #validation split results in 70/15/15 train/validation/test


score = model.evaluate(test_data, test_labels, batch_size=25)
print(score[0],score[1])


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")





# 1 0  = no motif, # 0 1 equals motif

freq_matrix = dict()

# if motif count present of each node

#why this works: tests for no motifs show a maximum of 2 similar 12 base pair strings while looping over 66750 constalations.
k = 750
test = 0
for j in test_labels:
    if j[0] == 1:
        continue
    else:
        for i in range(0,89):
            tempstring = ""
            for t in range(0,12):
                if test_data[k][i+t][0] == 1:
                    tempstring += "a"
                elif test_data[k][i+t][1] == 1:
                    tempstring +="c"
                elif test_data[k][i+t][2] == 1:
                    tempstring += "g"
                elif test_data[k][i+t][3] == 1:
                    tempstring +="t"
                else:
                    print("bug")
            if tempstring not in freq_matrix:
                freq_matrix[tempstring] = 1
            else:
                freq_matrix[tempstring] += 1
    k += 1
#print(freq_matrix[len(freq_matrix)-10:len(freq_matrix)])

print(k)
maxkey = max(freq_matrix, key=freq_matrix.get)

print(len(freq_matrix))
print(maxkey)
print(freq_matrix.get(maxkey))
"""


tmpall = [np.argmax(i) for i in pred]
print('The 10 first test list examples have been predicted as the following:')
print(tmp)
print(pred[:10])
print(pred[pred.shape[0]-10:pred.shape[0]-1])

#tests are only on with motif test, fix this
wrong_guesses = 0
for i in range(pred.shape[0]):
    realvalue = 1
    if tmpall[i] != realvalue:
        wrong_guesses += 1
print(wrong_guesses, "out of", pred.shape[0])

"""
#print(data[0])
#print(data.shape)

## que
"""

"""