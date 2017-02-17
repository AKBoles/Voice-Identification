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
training_data = '/home/cc/Data/Dev-Clean-Full/'
training_seg = '/home/cc/Data/Segment-Two/'

X = pickle.load(open('/home/cc/Data/pickle_files/speaker40_2secX.p', 'rb'))
Y = pickle.load(open('/home/cc/Data/pickle_files/speaker40_2secY.p', 'rb'))
speakers = speech_data.get_speakers(training_seg)

# define the network and the model for training
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

net = tflearn.input_data(shape=[None, 20, 87])
net = tflearn.fully_connected(net, 256)
net = tflearn.fully_connected(net, 256)
net = tflearn.fully_connected(net, 256)
net = tflearn.fully_connected(net, 256)
net = tflearn.fully_connected(net, 256)
net = tflearn.fully_connected(net, 256)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, len(speakers), activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

# now train the model!
model = tflearn.DNN(net)
model.fit(X, Y, n_epoch=1000, show_metric=True, snapshot_step=1000, run_id='SpeakerRec')
