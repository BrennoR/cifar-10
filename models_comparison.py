import numpy as np
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

# Models
shallow_model = get_shallow_model(input_shape, num_classes)
shallow_model_batch_norm = get_shallow_model(input_shape, num_classes, batch_normalization=True)
medium_model = get_medium_model(input_shape, num_classes)
medium_model_batch_norm = get_medium_model(input_shape, num_classes, batch_normalization=True)
deep_model = get_deep_model(input_shape, num_classes)
deep_model_batch_norm = get_deep_model(input_shape, num_classes, batch_normalization=True)

models = [shallow_model, shallow_model_batch_norm, medium_model, medium_model_batch_norm,
          deep_model, deep_model_batch_norm]
model_titles = ['Shallow Model', 'Shallow Model w/ Batch Normalization', 'Medium Model',
                'Medium Model w/ Batch Normalization', 'Deep Model', 'Deep Model w/ Batch Normalization']

test_losses = []
test_accuracies = []


for model, title in zip(models, model_titles):
    opt = keras.optimizers.Adam(lr=0.001)

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

    plot_accuracy(history, title)

print("Best model: {}".format(model_titles[np.argmax(test_accuracies)]))
