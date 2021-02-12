import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.callbacks import Callback

import tensorflow as tf
import datetime


df = pd.read_csv("iris.csv")
train, test = train_test_split(df,test_size=0.2)
trainX = train.drop(['species'], axis=1)
trainY = train['species'].astype('category')
trainY = trainY.cat.codes
trainY = np_utils.to_categorical(trainY)

testX = test.drop(['species'], axis=1)
testY = test['species'].astype('category')
testY = testY.cat.codes
testY = np_utils.to_categorical(testY)

model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(trainX, trainY, epochs=150, batch_size=10, callbacks=[tensorboard_callback])

_, accuracy = model.evaluate(testX, testY)
print('Accuracy: %.2f' % (accuracy*100))
