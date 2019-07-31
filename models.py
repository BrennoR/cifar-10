import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D


def get_shallow_model(input_shape, num_classes, batch_normalization=False):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    if batch_normalization:
        model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    if batch_normalization:
        model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    if batch_normalization:
        model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def get_medium_model(input_shape, num_classes, batch_normalization=False):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    if batch_normalization:
        model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    if batch_normalization:
        model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    if batch_normalization:
        model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    if batch_normalization:
        model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def get_deep_model(input_shape, num_classes, batch_normalization=False):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    if batch_normalization:
        model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    if batch_normalization:
        model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    if batch_normalization:
        model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    if batch_normalization:
        model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(256, kernel_size=2, activation='relu'))
    if batch_normalization:
        model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model
