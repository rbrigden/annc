# HW0 of 11-364

## Objective

The original assignment set forth instructed students to implement a feedforward
artificial neural network (ANN) in a relatively low level language or framework.
Although higher level scripting languages such as Python and Lua have wrapped heavily
optimized libraries that perform the same functions, the goal of this assignment is to truly understand the theoretical underpinnings of feedforward neural networks by writing the routines
from scratch (almost).

## Implementation

Although the original assignment suggested first implementing the network in the Cython
language, I decided to move to straight C. I wanted to ensure that I would have the time to both implement and debug as well as experiment with improvements to the vanilla network. My implementation uses the GNU Science Library (GSL) to perform matrix operations and in some cases directly calls on the seminal BLAS library. The code is broken up into three discrete modules as follows,

__/network/network.h__ contains the core data structures and algorithms for the neural network. It also contains a set of activation functions and cost functions, as well as a set of matrix helper routines.

__/training/training.h__ contains the routines used for training with mini batches and evaluating the network on test data.

__/mnist/mnist.h__ provides a simple data loader for the MNIST data set that is both space efficient and optimizes for speed of sample retrieval by the caller.

## Work log

### Interim report. (two weeks into the project)

Tasks Accomplished

- Implemented a feedforward neural network datastructure in C using BLAS libraries for matrix operations
- Used sigmoid neuron layers and implemented feedforward and backpropagation algorithms.
- Built a custom C loader for both MNIST and CIFAR-10 data
- Implemented stochastic gradient descent with mini-batches

Roadblock: Network is not performing faster than Nielson's python implementation and maxing out at 80% accuracy. I took your note into account this morning and tried to debug my C code using a small subset of the  MNIST data but was unable to pinpoint the error. Perhaps this is because I have written a lot of messy memory optimizations that have muddied the logic of both the feedforward and backpropagation algorithms, and led to an error that is very hard to pinpoint. I have put around 15 hours into the code so far and I hope to get back on track with a new angle of attack.

Possible solution: This evening I have already implemented about 30% of the functionality of the 2000 line codebase I had written previously in Google's Go programming language. The language has C-like syntax and compiles to machine code with C-competetive performance. It is typed and also has its own memory management system that will help me focus on writing the algorithms and spend less time dealing with memory management, which was taking >50% of my time when I was building in C. I hope to catch up my Go implementation to the functional extent of my C codebase by this Friday and then begin the optimization process.

I hope to implement the following techniques for next Wednesday with both MNIST and CIFAR-10:

- momentum
- L1/L2 regularization
- Dropout
- k-fold cross validation
- ADAM

### Switch to Golang

After writing the interim report, I pivoted to refactoring the C code into Go, Google's more expressive, type safe and thread safe C-variant language. After spending the next 3 days rewriting
the code in Go and debugging it, I was both satisfied and disappointed. It was a relief not to have to deal with memory management struggles, although I was running into difficulties with performance as the network continually failed to converge to zero on the XOR problem. These issues persisted even after adjusting the hyper-parameters a great number of times. This was not the
dagger, however, as I believed that I could eventually tracked down the bug. Once I found my bug (mislabelling a local variable) the computation was extremely slow, taking almost 5 minutes to
accomplish a single epoch (training on 60,000 MNIST examples & testing on 10,000). For kicks I went back to my C code and spent an evening fixing my memory leak issue, which as it turned out also took care of the logical error in the code as well. At that point, my C network achieved more and in much less time (about 15 seconds per epoch on the same dataset). This promising result
led me to switch my focus back to the C codebase.

### Returning to C

By the time I switched back to C, I had around a week and a half left to improve my results on the dataset. At this stage I shifted to further optimizing my network by switching out the
cost function from mean squared error (MSE or quadratic) to cross-entropy cost. This change drastically improved convergence, as expected. I also implemented momentum based weight and bias updates with promising results as well. Whereas before momentum the objective function would fluctuate rather radically, the momentum update had the expected smoothing effect on the decent and
also allowed me to reach a higher evaluation accuracy on the test set (~80% by epoch 15, converging at ~83%). Although I had already experimented with hyper-parameters such as the learning rate, the size of the network, and the mini-batch size previously, I began to experiment again and found that increasing the mini-batch size to 1000 helped push my accuracy on the test set to ~85% by epoch 15, converging ~90%. The finally change I made was to add L2 regularization which did not affect the convergence of the objective function, as expected, but allowed me to make much faster and continuous improvement on the test set accuracy. Below is a sample run of the latest version network.

```

$ ./annc
Initializing network
woohoo, verified
Initializing set loader

Reading data header
Number of images: 60000
Image height: 28
Image width: 28

Reading label header
Number of labels: 60000
Initializing set loader

Reading data header
Number of images: 10000
Image height: 28
Image width: 28

Reading label header
Number of labels: 10000

Epochs: 100, Eta: 0.500000, MBS: 1000
Mu: 0.9
Lambda: 0.8


evaluating...
cost: 18.381462
Epoch: 0, accuracy 5398 / 10000

evaluating...
cost: 13.036965
Epoch: 1, accuracy 7492 / 10000

evaluating...
cost: 10.506585
Epoch: 2, accuracy 7818 / 10000

evaluating...
cost: 8.890011
Epoch: 3, accuracy 8353 / 10000

evaluating...
cost: 7.573247
Epoch: 4, accuracy 8284 / 10000

evaluating...
cost: 6.859835
Epoch: 5, accuracy 8252 / 10000

evaluating...
cost: 6.398229
Epoch: 6, accuracy 8583 / 10000

evaluating...
cost: 6.022650
Epoch: 7, accuracy 8438 / 10000

evaluating...
cost: 5.833660
Epoch: 8, accuracy 8573 / 10000

evaluating...
cost: 5.478785
Epoch: 9, accuracy 8747 / 10000

evaluating...
cost: 5.316432
Epoch: 10, accuracy 8785 / 10000

evaluating...
cost: 5.349133
Epoch: 11, accuracy 8764 / 10000

evaluating...
cost: 5.095236
Epoch: 12, accuracy 8749 / 10000

evaluating...
cost: 5.080301
Epoch: 13, accuracy 8726 / 10000

evaluating...
cost: 5.045305
Epoch: 14, accuracy 8748 / 10000

evaluating...
cost: 4.842447
Epoch: 15, accuracy 8755 / 10000

evaluating...
cost: 4.931275
Epoch: 16, accuracy 8788 / 10000

evaluating...
cost: 4.775373
Epoch: 17, accuracy 8832 / 10000

evaluating...
cost: 4.691695
Epoch: 18, accuracy 8743 / 10000

evaluating...
cost: 4.791714
Epoch: 19, accuracy 8686 / 10000

evaluating...
cost: 4.712357
Epoch: 20, accuracy 8806 / 10000

evaluating...
cost: 4.552620
Epoch: 21, accuracy 8856 / 10000

evaluating...
cost: 4.444064
Epoch: 22, accuracy 8916 / 10000

evaluating...
cost: 4.412344
Epoch: 23, accuracy 8916 / 10000

evaluating...
cost: 4.488347
Epoch: 24, accuracy 8856 / 10000

evaluating...
cost: 4.436550
Epoch: 25, accuracy 8940 / 10000

evaluating...
cost: 4.195865
Epoch: 26, accuracy 8977 / 10000

evaluating...
cost: 4.104246
Epoch: 27, accuracy 8951 / 10000

evaluating...
cost: 4.237411
Epoch: 28, accuracy 8946 / 10000

evaluating...
cost: 4.310728
Epoch: 29, accuracy 8946 / 10000

evaluating...
cost: 4.265719
Epoch: 30, accuracy 8989 / 10000

...

evaluating...
cost: 3.646227
Epoch: 92, accuracy 9069 / 10000

evaluating...
cost: 3.626412
Epoch: 93, accuracy 9088 / 10000

evaluating...
cost: 3.598544
Epoch: 94, accuracy 9153 / 10000

evaluating...
cost: 3.520385
Epoch: 95, accuracy 9114 / 10000

evaluating...
cost: 3.798848
Epoch: 96, accuracy 9031 / 10000

evaluating...
cost: 3.768557
Epoch: 97, accuracy 9029 / 10000

evaluating...
cost: 3.743984
Epoch: 98, accuracy 9107 / 10000

evaluating...
cost: 3.702165
Epoch: 99, accuracy 9091 / 10000

```
