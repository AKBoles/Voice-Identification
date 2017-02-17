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
training_data = '/home/cc/Data/Full-100/'
training_seg = '/home/cc/Data/Segment-100-Two/'

# calculate the mfcc matrices for training from the segmented data
#X = []
#Y = []
speakers = speech_data.get_speakers(training_seg)
#for f in os.listdir(training_seg):
#    Y.append(speech_data.one_hot_from_item(speech_data.speaker(f), speakers))
#    y, sr = librosa.load(training_seg + f)
#    X.append(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20))
#pickle.dump(X, open('/home/cc/Data/pickle_files/speaker250_2secX.p', 'wb'))
#pickle.dump(Y, open('/home/cc/Data/pickle_files/speaker250_2secY.p', 'wb'))
X = pickle.load(open('/home/cc/Data/pickle_files/speaker250_2secX.p', 'rb'))
Y = pickle.load(open('/home/cc/Data/pickle_files/speaker250_2secY.p', 'rb'))

# define the network and the model for training
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

net = tflearn.input_data(shape=[None, 20, 87])
net = tflearn.fully_connected(net, 512)
net = tflearn.fully_connected(net, 512)
net = tflearn.fully_connected(net, 512)
net = tflearn.fully_connected(net, 512)
net = tflearn.fully_connected(net, 512)
net = tflearn.fully_connected(net, 512)
net = tflearn.fully_connected(net, 512)
net = tflearn.fully_connected(net, 512)
net = tflearn.fully_connected(net, 512)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, len(speakers), activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

# now train the model!
model = tflearn.DNN(net)
model.fit(X, Y, n_epoch=1000, show_metric=True, snapshot_step=1000, run_id='SpeakerRec')
