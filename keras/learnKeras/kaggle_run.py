from kaggle_data import load_data,preprocess_data,preprocess_labels
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

x_train, labels = load_data('../data/kaggle_ottogroup/train.csv', train=True)
x_train, scaler = preprocess_data(x_train)  #特征标准化
y_train, encoder = preprocess_labels(labels)

x_test, ids = load_data('../data/kaggle_ottogroup/test.csv', train=False)
x_test, _ = preprocess_labels(x_test, scaler)

nb_classes = y_train.shape[1]

print(x_train.shape)    # (61878, 93)
print(y_train.shape)    #(61878, 9)
print(nb_classes, 'classes')  # 9 classes

dims = x_train.shape[1]
print(dims, 'dims')   # 93 dims

print('Building model...')

model = Sequential()
model.add(Dense(nb_classes, input_shape=(dims,), activation='sigmoid'))
model.add(Activation('softmax'))

model.compile(optimizer='sgd', loss='categorical_crossentropy')
model.fit(x_train, y_train)
model.summary()