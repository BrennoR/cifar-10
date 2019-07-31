import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from keras.datasets import cifar10

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

# CIFAR-10 labels
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Plots
plt.figure()
plt.suptitle('CIFAR-10 Dataset')
for i in range(8):
    ax = plt.subplot(240 + i + 1)
    ax.set_title(labels[y_train[i][0]])
    plt.imshow(X_train[i])
plt.show()
