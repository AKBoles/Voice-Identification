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
training = '/home/cc/Data/Dev-Clean-Train-Two/'
testing = '/home/cc/Data/Dev-Clean-Test-Two/'

# load the mfcc matrices for training from pickled data
X = pickle.load(open('/home/cc/Data/pickle_files/devfull_2secX.p', 'rb'))
Y = pickle.load(open('/home/cc/Data/pickle_files/devfull_2secY.p', 'rb'))
speakers = speech_data.get_speakers(training)

# input size for fully connected layers
layer_size = int(sys.argv[1])
dropout = float(sys.argv[2])

# define the network and the model for training
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

# for just mfcc
net = tflearn.input_data(shape=[None, 20, 87])
net = tflearn.fully_connected(net, layer_size)
net = tflearn.fully_connected(net, layer_size)
net = tflearn.fully_connected(net, layer_size)
net = tflearn.dropout(net, dropout)
net = tflearn.fully_connected(net, len(speakers), activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

# now train the model!
t0 = time.time()
model = tflearn.DNN(net)
model.fit(X, Y, n_epoch=100, show_metric=True, snapshot_step=1000, validation_set=0.1)
t1 = time.time()

# load the test mfcc values for testing from pickled data
Xtest = pickle.load(open('/home/cc/Data/pickle_files/devfull_2sectestX.p', 'rb'))
Ytest = pickle.load(open('/home/cc/Data/pickle_files/devfull_2sectestY.p', 'rb'))

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

with open('libri_3layer_stats.txt', 'a') as myfile:
    [myfile.write(a + ' , ') for a in l]
    myfile.write('\n')
