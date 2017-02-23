# annc

## Objective

Implement a feedforward artificial neural network (ANN) in a relatively low level language or framework.
Although higher level scripting languages such as Python and Lua have wrapped heavily
optimized libraries that perform the same functions, it is important to implement these
core routines from scratch to truly understand the theoretical underpinnings of feedforward
neural networks.

## Implementation

This implementation uses the GNU Science Library (GSL) to perform matrix operations and in some cases directly calls on the seminal BLAS library. The code is broken up into three discrete modules as follows,

__/network/network.h__ contains the core data structures and algorithms for the neural network. It also contains a set of activation functions and cost functions, as well as a set of matrix helper routines.

__/training/training.h__ contains the routines used for training with mini batches and evaluating the network on test data.

__/mnist/mnist.h__ provides a simple data loader for the MNIST data set that is both space efficient and optimizes for speed of sample retrieval by the caller.

## Sample training

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
