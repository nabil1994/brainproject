import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import linear_model, datasets
from sklearn import preprocessing
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
from scipy import signal
import scipy.io
import math
from sklearn.neural_network import BernoulliRBM


# sigmoid function
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))
# load data
train_x = np.array(scipy.io.loadmat("train_x.mat")['train_x'])
train_y = np.array(scipy.io.loadmat("train_y.mat")['train_y'])
testx = np.array(scipy.io.loadmat("test_x.mat")['test_x'])
testy = np.array(scipy.io.loadmat("test_y.mat")['test_y'])
testx.reshape((1500,70))

rbm = BernoulliRBM(n_components = 200, n_iter = 40,learning_rate = 0.01,  verbose = True)
rbm.fit(train_x)
np.save("rbm_weights.npy",rbm.components_)
# print(trainy)
W = np.load("rbm_weights.npy")
print(W.shape)
# matplotlib inline
weightmap_shape = (30, 50)
np.save("rbm_biases.npy",rbm.intercept_hidden_)
np.save("hidden.npy",rbm.transform(train_x))

rbm_biases = np.load("rbm_biases.npy")
rbm_weights = np.load("rbm_weights.npy")
print(rbm_biases.shape, rbm_weights.shape)
rbm_weight = rbm_weights.T
rbm_bias = rbm_biases
#
model = Sequential()
model.add(Dense(200, activation='relu', input_dim=1500, name='rbm'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
layer = model.get_layer('rbm')
layer.set_weights([rbm_weight,rbm_bias])
model.fit(train_x, train_y, epochs=10, batch_size=50)
print(classification_report(testy, np.argmax(model.predict(testx),axis=1)))
