import numpy as np
from keras.utils import np_utils
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split
"""
the function load_data() returns a tuple of all_data,test_data,test_labels constructed with the same seed as in train_model.py
to be used when evaluating the model and constructing the position weight matrix (pwm).
"""


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

def load_data():
    print("load_data returns the following tuple:")
    print("all_data,test_data,test_labels")
    model_choice = "model1.h5"
    if model_choice == "model1.h5":
        f0 = open("dna_5000_100_0_[].fa", 'r')
        f1 = open("dna_5000_100_1_['TAAAGCGTAATA'].fa", 'r')
    if model_choice == "model2.h5":
        f0 = open("seq2_dna_5000_100_0_[].fa",'r')
        f1 = open("seq2_dna_5000_100_1_['CTCTTGAGG'].fa",'r')

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
    all_data = np.concatenate((data,test_data),axis=0)

    if model_choice == "model1.h5":
        test_data.tofile("testdata1.txt")
    if model_choice == "model2.h5":
        test_data.tofile("testdata2.txt")
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

    #if you update the return values, update the printed instructions accordingly!!!
    return all_data,test_data,test_labels