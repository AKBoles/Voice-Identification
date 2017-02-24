import tflearn
import librosa
import pannous_speech_data as data

learning_rate = 0.0001
training_iters = 300000

batch = word_batch = data.mfcc_batch_generator(64)
X, Y = next(batch)
trainX, trainY = X,Y
testX, testY = X, Y

# define network
net = tflearn.input_data([None, 20, 80])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 10, activation = 'softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=3)

while 1:
    model.fit(trainX, trainY, n_epoch=5000, validation_set=(testX, testY), show_metric=True, batch_size=64)
    _y = model.predict(X)
