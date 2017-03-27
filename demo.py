import os
import sys
import librosa
import pydub
import pyaudio
import tflearn
# change this directory to: Data/modules/
sys.path.append('data_prep/')
import speech_data
import segment_data
import record_audio

# constants - directory, etc.
# imput these according to operating system
training_dir = 'Data/train/audio/'
training_seg = 'Data/train/segment/'
testing_dir = 'Data/test/audio/'
testing_seg = 'Data/test/segment/'
model_dir = 'Data/model/'

# record audio for training, specify how many speakers and files per speaker with command line
num_speakers = int(sys.argv[1])
num_files = int(sys.argv[2])
count = 0
while count < num_speakers:
    # record audio for each speaker - input strings for their names later
    for s in range(num_files):
        # record audio, save to a file
        raw_input('Press Enter to record an audio file.')
        record_audio.record_to_file(training_dir + 'Speaker' + str(count) + '_' + str(s) + '.wav')
    count = count + 1

# train model over training directory, make sure to segment data
length = 1
if not os.listdir(training_seg):
    segment_data.segment(data=training_dir, seg_location=training_seg, length=length)

# create mfcc, delta, delta2 input features
hop_length = 128 # default = 512
X = []
Y = []
speakers = speech_data.get_speakers(training_dir)
for f in os.listdir(training_seg):
    Y.append(speech_data.one_hot_from_item(speech_data.speaker(f), speakers))
    y, sr = librosa.load(training_seg + f)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    X.append(np.c_[mfcc,delta,delta2])

# define the network for training
layer_size = 128
dropout = 0.7
learning_rate = 0.001
tflearn.init_graph(num_cores=8,gpu_memory_fraction=0.5)

net = tflearn.input_data(shape=[None, 20, 519])
net = tflearn.fully_connected(net, layer_size, activation='relu')
net = tflearn.fully_connected(net, layer_size, activation='relu')
net = tflearn.fully_connected(net, layer_size, activation='relu')
net = tflearn.dropout(net, dropout)
net = tflearn.fully_connected(net, len(speakers), activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy', learning_rate=learning_rate)

# now train the model!
run_id='Demonstration'
model = tflearn.DNN(net)
model.fit(X, Y, n_epoch=50, show_metric=True, snapshot_step=1000, run_id=run_id, validation_set=0.1)

# save the model to model_dir
model.save(model_dir + 'demo.model')

# record for testing model - for now just doing one speaker test file, later create another python script for this
# record audio, save to a file
raw_input('Press Enter to record an audio file.')
record_audio.record_to_file(testing_dir + 'test.wav')

# segment the recorded test audio
if not os.listdir(testing_seg):
    segment_data.segment(data=testing_dir, seg_location=testing_seg, length=length)

# create mfcc, delta, delta2 for testing data
Xtest = []
for f in os.listdir(testing_seg):
    y, sr = librosa.load(testing_seg + f)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    Xtest.append(np.c_[mfcc,delta,delta2])

result = model.predict(Xtest)
c = 0
for f,r in zip(os.listdir(testing_seg), result):
    res = speech_data.one_hot_to_item(r, speakers)
    if res in f:
        c = c + 1
acc = float(c) / float(len(Xtest))
print('Test set accuracy: %s' %str(acc))

# need to add a part that outputs the speaker to command line
