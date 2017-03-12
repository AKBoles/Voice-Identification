import os
import sys
import time
import librosa
import tflearn
import pydub
import wave
import pickle
import speech_data
import segment_data
import tensorflow as tf
import librosa.display
import numpy as np

# load constants - training directory, testing directory
training = '/home/cc/Data/train/'
testing = '/home/cc/Data/test/'

# calculate the mfcc matrices for training from the segmented data
X = []
Y = []
speakers = speech_data.get_speakers(training)
for f in os.listdir(training):
    Y.append(speech_data.one_hot_from_item(speech_data.speaker(f), speakers))
    y, sr = librosa.load(training + f)
    mfcc = np.asarray(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20))
    X.append(mfcc)

# input size for fully connected layers
layer_size = int(sys.argv[1])
dropout = float(sys.argv[2])

# define the network and the model for training
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

# for just mfcc
net = tflearn.input_data(shape=[None, 20, 87])
net = tflearn.fully_connected(net, layer_size)
net = tflearn.fully_connected(net, layer_size)
#net = tflearn.fully_connected(net, layer_size)
net = tflearn.dropout(net, dropout)
net = tflearn.fully_connected(net, len(speakers), activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

# now train the model!
t0 = time.time()
model = tflearn.DNN(net)
model.fit(X, Y, n_epoch=100, show_metric=True, snapshot_step=1000, validation_set=0.05)
t1 = time.time()

# test the trained model using the testing directory
# calculate the mfcc matrices for testing from the segmented data
Xtest = []
Ytest = []
speakers = speech_data.get_speakers(testing)
for f in os.listdir(testing):
    Ytest.append(speech_data.one_hot_from_item(speech_data.speaker(f), speakers))
    y, sr = librosa.load(testing + f)
    Xtest.append(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20))

# now test model over the test segments
result = model.predict(Xtest)
c = 0
for f,r in zip(os.listdir(testing), result):
    res = speech_data.one_hot_to_item(r, speakers)
    if res in f:
        c = c + 1
acc = float(c) / float(len(Xtest))

# now output to a text file for comparison
l = ['Layer Size : ' + str(layer_size), 'Dropout: ' + str(dropout), 'Test Acc: ' + str(acc), 'Train time: ' + str(t1 - t0)]

with open('oci_2layer_stats.txt', 'a') as myfile:
    [myfile.write(a + ' , ') for a in l]
    myfile.write('\n')
