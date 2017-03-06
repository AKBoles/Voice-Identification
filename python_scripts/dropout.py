import os
import sys
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
training_seg = '/home/cc/Data/Dev-Clean-Train-Two/'
testing = '/home/cc/Data/Dev-Clean-Test-Two'

# size of fully connected layers
n = sys.argv[1]
d = sys.argv[2]
m = 18

# calculate the mfcc matrices for training from the segmented data
#X = []
#Y = []
speakers = speech_data.get_speakers(training_seg)
#for f in os.listdir(training_seg):
#    Y.append(speech_data.one_hot_from_item(speech_data.speaker(f), speakers))
#    y, sr = librosa.load(training_seg + f)
#    X.append(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=int(m)))

#pickle.dump(X, open('/home/cc/Data/pickle_files/mfcc_len/train' + str(m) + '_X.p', 'wb'))
#pickle.dump(Y, open('/home/cc/Data/pickle_files/mfcc_len/train' + str(m) + '_Y.p', 'wb'))
X = pickle.load(open('/home/cc/Data/pickle_files/mfcc_len/train' + str(m) + '_X.p', 'rb'))
Y = pickle.load(open('/home/cc/Data/pickle_files/mfcc_len/train' + str(m) + '_Y.p', 'rb'))

# define the network and the model for training
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

net = tflearn.input_data(shape=[None, int(m), 87])
net = tflearn.fully_connected(net, n)
net = tflearn.fully_connected(net, n)
net = tflearn.fully_connected(net, n)
net = tflearn.fully_connected(net, n)
net = tflearn.fully_connected(net, n)
net = tflearn.fully_connected(net, n)
#net = tflearn.dropout(net, 0.8)
#net = tflearn.fully_connected(net, n)
#net = tflearn.fully_connected(net, n)
#net = tflearn.fully_connected(net, n)
net = tflearn.dropout(net, float(d))
net = tflearn.fully_connected(net, len(speakers), activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

# now train the model!
model = tflearn.DNN(net)
model.fit(X, Y, n_epoch=1000, show_metric=True, snapshot_step=1000, run_id='DropoutAnalysis-add-' + str(d))

# test the model

