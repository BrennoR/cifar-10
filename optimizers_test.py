import numpy as np
import seaborn as sns
sns.set()

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

from models import get_shallow_model
from utilities import plot_multiple_accs

# Parameters
batch_size = 32
num_classes = 10
epochs = 100

# Data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255  # normalize train and test data
X_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

input_shape = X_train.shape[1:]

adam = keras.optimizers.Adam()
sgd = keras.optimizers.SGD()
rmsprop = keras.optimizers.RMSprop()
adagrad = keras.optimizers.Adagrad()
adadelta = keras.optimizers.Adadelta()
adamax = keras.optimizers.Adamax()
nadam = keras.optimizers.Nadam()

opts = [adam, sgd, rmsprop, adagrad, adadelta, adamax, nadam]
opt_labels = ['Adam', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam']

test_losses = []
test_accuracies = []

histories = []

for opt, title in zip(opts, opt_labels):
    model = get_shallow_model(input_shape, num_classes, batch_normalization=True)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    datagen = ImageDataGenerator()
    datagen.fit(X_train)

    STEP_SIZE_TRAIN = X_train.shape[0] // batch_size

    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                                  epochs=epochs,
                                  steps_per_epoch=STEP_SIZE_TRAIN,
                                  validation_data=(X_test, y_test),
                                  )

    scores = model.evaluate(X_test, y_test, verbose=1)

    test_losses.append(scores[0])
    test_accuracies.append(scores[1])

    histories.append(history)

plot_multiple_accs(histories, 'Accuracy of Various Optimizers', opt_labels)

print("Best model: {}, Best accuracy: {}".format(opt_labels[np.argmax(test_accuracies)], max(test_accuracies)))
