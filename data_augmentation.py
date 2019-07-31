import seaborn as sns
sns.set()

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

from models import get_shallow_model, get_medium_model, get_deep_model
from utilities import plot_accuracy

# Parameters
batch_size = 32
num_classes = 10
epochs = 500

# Data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255  # normalize train and test data
X_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

input_shape = X_train.shape[1:]

model = get_deep_model(input_shape, num_classes, batch_normalization=True)

opt = keras.optimizers.Adamax()

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

datagen = ImageDataGenerator(width_shift_range=0.2,
                             height_shift_range=0.2,
                             rotation_range=0.2,
                             vertical_flip=True,
                             horizontal_flip=True)

datagen.fit(X_train)

STEP_SIZE_TRAIN = X_train.shape[0] // batch_size

history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                              epochs=epochs,
                              steps_per_epoch=STEP_SIZE_TRAIN,
                              validation_data=(X_test, y_test),
                              )

scores = model.evaluate(X_test, y_test, verbose=1)

print("Test Loss:", scores[0])
print("Test Accuracy:", scores[1])

plot_accuracy(history, 'Deep Model w/ Batch Normalization and Data Augmentation')

