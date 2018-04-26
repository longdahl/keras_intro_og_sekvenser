import numpy as np
import load_data as ld
from keras.models import Model,load_model

all_data,test_data,test_labels = ld.load_data()
model = load_model("model1.h5")
np.random.shuffle(all_data)

pred = model.predict(all_data,batch_size=32)
tmp = [np.argmax(i) for i in pred]

#specifying the layer we want outputted in our intermediate_layer_model
layer_name = "conv2d_1"


#creating our intermediate layer model, from our trained_model
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)

#calculating the values of the nodes in the specified (layer_name) hidden layer
intermediate_output = intermediate_layer_model.predict(all_data)

startindex_list = []
pfm = np.zeros((4,12))
string_count = 0
skip_count = 0

intermediate_output = intermediate_output.reshape(10000,89,64,1)
for string_conv in intermediate_output:
    max = 0
    pos = 0
    # for each dna_string we search for the position over all convolutions and kernels with the maximum value
    #the index of the node in the convolutional layer is the start_index of the binding sequence
    for filter in string_conv:
        for value in filter:
            if value > max:
                max = value
                maxpos = pos
        pos += 1
    c = 0
    if tmp[string_count] == 1: #we only look at cases where we predict a binding motif
        #from these we construct the position frequency matrix by summing base pair occurences
        for OneHot in all_data[string_count][maxpos:maxpos+12]:
            if OneHot[0] == 1:
                pfm[0][c] += 1
            elif OneHot[1] == 1:
                pfm[1][c] += 1
            elif OneHot[2] == 1:
                pfm[2][c] += 1
            elif OneHot[3] == 1:
                pfm[3][c] += 1
            c += 1
    string_count += 1

pwm = np.zeros((4,12))

#converting position frequency matrix to position weight matrix
for col in range(0,12):
    sum = pfm[0][col] + pfm[1][col] + pfm[2][col] + pfm[3][col]

    pwm[0][col] = pfm[0][col] / sum
    pwm[1][col] = pfm[1][col] / sum
    pwm[2][col] = pfm[2][col] / sum
    pwm[3][col] = pfm[3][col] / sum

pwm_rounded = np.round_(pwm,2)

print(pwm_rounded)

#saving model (already saved)
#np.savetxt("pwm.csv",pwm, delimiter=",")