import numpy as np
from keras.utils import np_utils
test_data_size = 1500



test_no_motif_label = np.zeros((int(test_data_size/2),1))
test_with_motif_label = np.ones((int(test_data_size/2),1))

test_labels = np.concatenate((test_no_motif_label,test_with_motif_label), axis=0)

test_labels = np_utils.to_categorical(list(test_labels))

f1 = open("dna_5000_100_1_['TAAAGCGTAATA'].fa",'r')
freq_matrix = dict()

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


with_motif_data = create_matrix(f1)

with_motif_data = np_utils.to_categorical(list(with_motif_data))
with_motif_data = with_motif_data.reshape(5000,100,4,1)


print(with_motif_data.shape)


with_motif_data = with_motif_data[:1500]

print(with_motif_data.shape)
t = 0
temp_list = []
for dnastring in with_motif_data:
    tempstring = ""
    for index in dnastring:
        if index[0] == 1:
            tempstring += "a"
        elif index[1] == 1:
            tempstring += "c"
        elif index[2] == 1:
            tempstring += "g"
        elif index[3] == 1:
            tempstring += "t"
    temp_list.append(tempstring)



print(temp_list[0] == "gtgcggacggtggactgtataaagcctcatagctggaacggtaccaggcctcagctgtaacactccccgtgttgaaactcgtgcgtggtgaggtcgcctt")
