# cifar-10
Repository of a study on using different model architectures, optimizers, and techniques to classify the CIFAR-10 dataset. Three different
model architectures of varying sizes were tested (shallow, medium, and deep). Seven optimizers were compared (Adam, SGD, RMSprop, Adamax,
Adadelta, Adagrad, and Nadam). Finally the effects of batch normalization and data augmentation on the final accuracy were observed. 
The best model utilized the deep architecture, Adamax optimizer, batch normalization, and data augmentation to achieve a final test 
accuracy of 84%.

## Dataset
The CIFAR-10 dataset consists of 50,000 training images and 10,000 test images. The images are color and 32x32 pixels in size. There are
10 total classes in the dataset with 6,000 images per class. The classes are:
* airplane
* automobile
* bird
* cat
* deer
* dog
* frog
* horse
* ship
* truck

A preview of the dataset is shown below:
\
\
![alt text](https://github.com/BrennoR/cifar-10/blob/master/plots/cifar10_visualization.PNG "Dataset Visualization")

## Model Architectures
Three different model architectures of varying sizes were tested on the dataset. A shallow, a medium, and a deep convolutional 
neural network architecture. These models are shown below:

### Shallow Model

| Layer         | Filters       |    Size       |   Kernel Size |   Stride      | Activation    |
| :-----------: | :-----------: | :-----------: | :-----------: | :----------:  | :-----------: |
|   Input | 1 | 32x32 | - | - | - |
| Conv2D | 32 | 30x30 | 3x3 | 1 | ReLU |
| Conv2D | 32 | 28x28 | 3x3 | 1 | ReLU |
| MaxPooling2D | 32 | 14x14 | 2x2 | 2 | ReLU |
| Conv2D | 64 | 12x12 | 3x3 | 1 | ReLU |
| MaxPooling2D | 64 | 6x6 | 2x2 | 2 | ReLU |
| Conv2D| 128 | 4x4 | 3x3 | 1 | ReLU |
| MaxPooling2D | 128 | 2x2 | 2x2 | 2 | ReLU |
| Dense  |  - | 512 | - | - | ReLU |
| Dropout | -  | 0.5 | - | - | - |
| Dense  |  - | 512 | - | - | ReLU |
| Dropout | -  | 0.5 | - | - | - |
| Dense | - | 10 | - | - | Softmax | 


### Medium Model

| Layer         | Filters       |    Size       |   Kernel Size |   Stride      | Activation    |
| :-----------: | :-----------: | :-----------: | :-----------: | :----------:  | :-----------: |
|   Input | 1 | 32x32 | - | - | - |
| Conv2D | 32 | 30x30 | 3x3 | 1 | ReLU |
| Conv2D | 32 | 28x28 | 3x3 | 1 | ReLU |
| MaxPooling2D | 32 | 14x14 | 2x2 | 2 | ReLU |
| Conv2D | 64 | 12x12 | 3x3 | 1 | ReLU |
| Conv2D | 64 | 10x10 | 3x3 | 1 | ReLU |
| Conv2D| 128 | 8x8 | 3x3 | 1 | ReLU |
| MaxPooling2D | 128 | 4x4 | 2x2 | 2 | ReLU |
| Conv2D| 128 | 2x2 | 3x3 | 1 | ReLU |
| Dense  |  - | 512 | - | - | ReLU |
| Dropout | -  | 0.5 | - | - | - |
| Dense  |  - | 512 | - | - | ReLU |
| Dropout | -  | 0.5 | - | - | - |
| Dense | - | 10 | - | - | Softmax | 

### Deep Model

| Layer         | Filters       |    Size       |   Kernel Size |   Stride      | Activation    |
| :-----------: | :-----------: | :-----------: | :-----------: | :----------:  | :-----------: |
|   Input | 1 | 32x32 | - | - | - |
| Conv2D | 32 | 30x30 | 3x3 | 1 | ReLU |
| Conv2D | 32 | 28x28 | 3x3 | 1 | ReLU |
| MaxPooling2D | 32 | 14x14 | 2x2 | 2 | ReLU |
| Conv2D | 64 | 12x12 | 3x3 | 1 | ReLU |
| Conv2D | 64 | 10x10 | 3x3 | 1 | ReLU |
| Conv2D| 128 | 8x8 | 3x3 | 1 | ReLU |
| Conv2D| 128 | 6x6 | 3x3 | 1 | ReLU |
| Conv2D| 256 | 4x4 | 3x3 | 1 | ReLU |
| MaxPooling2D |256 | 2x2   | 2x2 | 2 | ReLU |
| Conv2D| 256 | 1x1 | 2x2 | 1 | ReLU |
| Dense  |  - | 512 | - | - | ReLU |
| Dropout | -  | 0.5 | - | - | - |
| Dense  |  - | 512 | - | - | ReLU |
| Dropout | -  | 0.5 | - | - | - |
| Dense | - | 10 | - | - | Softmax |

<br/>
<br/>
All models were built using the Keras library. The three models were initially tested using the Adam optimizer with a learning rate of
0.001 for 100 epochs. The results are shown below:
<br/>
<br/>

![alt text](https://github.com/BrennoR/cifar-10/blob/master/plots/cifar10_shallow.PNG "Shallow Model")
![alt text](https://github.com/BrennoR/cifar-10/blob/master/plots/cifar10_medium.PNG "Medium Model")
![alt text](https://github.com/BrennoR/cifar-10/blob/master/plots/cifar10_deep.PNG "Deep Model")

As shown above, the shallow model actually performed the best out of the three reaching a test accuracy of 74%. All models started
to severly overfit after just a couple epochs. Interestingly, the training accuracy of the deep model started to decrease after 60
epochs.

## Batch Normalization
Batch normalization layers were added to various locations in the different model architectures. This additional regularization was
added to combat the overfitting present in the initial tests. The results are shown below:

![alt text](https://github.com/BrennoR/cifar-10/blob/master/plots/cifar10_shallow_batch.PNG "Shallow Model w/ Batch Normalization")
![alt text](https://github.com/BrennoR/cifar-10/blob/master/plots/cifar10_medium_batch.PNG "Medium Model w/ Batch Normalization")
![alt text](https://github.com/BrennoR/cifar-10/blob/master/plots/cifar10_deep_batch.PNG "Deep Model w/ Batch Normalization")

As shown above, all models performed decently better with batch normalization. The models attained final test accuracies of around
77 to 79 %. The shallow model gained a 5 % improvement with batch normalization and once again performed the best out of the trio.

## Optimizers Comparison
Seven optimizers were tested on the shallow model with batch normalization in order to compare their respective performances. The
default parameters (learning rate, betas, etc...) provided by Keras were used for each optimizer. The results obtained after 100
epochs are shown below:

![alt text](https://github.com/BrennoR/cifar-10/blob/master/plots/cifar10_optimizers.PNG "Optimizers Comparison")

As shown above, most optimizers obtained fairly similar results with Adam and Adamax performing the best. RMSprop had a very irregular
learning curve. This curve is most likely explained by the use of the default parameters. It is possible that RMSprop would perform as
well as the other optimizers given some tuning. Due to being the best performer, Adamax was chosen as the optimizer to be used for
subsequent tests.

## Data Augmentation
The last technique studied was data augmentation. Keras' ImageDataGenerator class was used to augment the original CIFAR-10 dataset.
The four augmentation parameters set and their respective values were:
* width_shift_range = 0.2
* height_shift_range = 0.2
* horizontal_flip = True
* vertical_flip = True

The first test involved using the shallow model with batch normalization. The results are shown below:

![alt text](https://github.com/BrennoR/cifar-10/blob/master/plots/cifar10_shallow_batch_augmentation.PNG "Shallow Model w/ Batch Normalization and Data Augmentation")

As shown above, data augmentation did indeed improve the results of the shallow model which reached a maximum accuracy of about 80
to 81 %. The model also did not exhibit any signs of overfitting even after 100 epochs.\
\
Data augmentation was then used with the deep model with batch normalization. The deep model suffered from severe overfitting in
previous tests and data augmentation is an excellent tool to combat overfitting and to help the model learn. The results from this test
are shown below:

![alt text](https://github.com/BrennoR/cifar-10/blob/master/plots/cifar10_deep_batch_augmentation.PNG "Deep Model w/ Batch Normalization and Data Augmentation")

As shown above, the deep model with batch normalization and data augmentation performed the best out of all previous models. It reached
a final test accuracy of 84%! Subsequent tests were made using additional data augmentation techniques such as scaling, shearing, and
rotation. Surprisingly, these tests were shown to actually hurt the overall performance of the model.

## Conclusions
There are many conclusions to be drawn from this study. The different model architectures demonstrated that it is extremely easy to
overfit this dataset. All model sizes began to overfit after only a couple of epochs. Batch normalization helped reduce overfitting and
led to greater accuracies for all architectures (5% improvement for the shallow model). The various optimizers performed similarly, but
Adam and Adamax were the best performers. It is very possible however that the others could perform just as well given hyperparameter 
tweaking. This is an area for further study. Data augmentation had an immense effect on the accuracy of the models. Due to a very effective
reduction in overfitting, deeper models could be used in conjunction with data augmentation to achieve a higher final accuracy. 

## Future Work
There is much future work to be done on different model architectures, sizes, and techniques. Many more tests can be run on data augmentation
techniques in order to further improve performance. In addition, other techniques such as kernel regularization and their effect on
performance can be tested. 
