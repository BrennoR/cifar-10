import matplotlib.pyplot as plt


# plots epochs vs. accuracies, train and test
def plot_accuracy(history, title):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'])
    plt.show()


# plots epochs vs. accuracy for multiple histories, test only
def plot_multiple_accs(histories, title, opt_labels):
    for history in histories:
        plt.plot(history.history['val_acc'])
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(opt_labels)
    plt.show()
