import load_data as ld
from keras.models import load_model


#loading the test data
all_data,test_data,test_labels = ld.load_data()

#loading our model trained with train_model.py
model = load_model("model1.h5")

#evaluating our model
score = model.evaluate(test_data, test_labels, batch_size=25)
print("loss           accurracy")
print(score[0],score[1])



