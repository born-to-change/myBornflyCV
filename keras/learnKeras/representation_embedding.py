import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


nb_classes = 10
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

# Put everything on grayscale
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train)  #Split arrays or matrices into random train and test subsets
# train_data：所要划分的样本特征集  train_target：所要划分的样本结果
#test_size:default=0.25 random_state(随机数种子):但填0或不填，每次都会不一样

# plt.imshow(X_train[0].reshape(28, 28))
# plt.show()

# train
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001),
              metrics=['accuracy'])

network_history = model.fit(X_train, Y_train, batch_size=128,
                            shuffle=True, epochs=5, verbose=1,
                            validation_data=(X_val, Y_val))

def plot_history(network_history):
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(network_history.history['loss'])
    plt.plot(network_history.history['val_loss'])
    plt.legend(['Training', 'Validation'])

    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(network_history.history['acc'])
    plt.plot(network_history.history['val_acc'])
    plt.legend(['Training', 'Validation'], loc='low right')
    plt.show()

plot_history(network_history)