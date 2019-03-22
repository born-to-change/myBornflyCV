import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPool2D, Dense, Dropout, Flatten

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)  # (60000, 28, 28)
print(y_train.shape)  # (60000,)

batch_size = 128
num_classes = 10
epochs = 7

x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(filters=32,
                 kernel_size=(3, 3),
                 data_format='channels_first',
                 activation='relu',
                 input_shape=(1, 28, 28)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
#pool_size: 整数，最大池化的窗口大小。
#strides: 整数，或者是 None。作为缩小比例的因数。 例如，2 会使得输入张量缩小一半。 如果是 None，那么默认值是 pool_size
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer=keras.optimizers.rmsprop(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('accuracy:', score[1])
print('loss', score[0])