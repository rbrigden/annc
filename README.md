# HW0 of 11-364

## Objective

The original assignment set forth instructed students to implement a feedforward
artificial neural network (ANN) in a relatively low level language or framework.
Although higher level scripting languages such as Python and Lua have wrapped heavily
optimized libraries that perform the same functions, the goal of this assignment is to truly understand the theoretical underpinnings of feedforward neural networks by writing the routines
from scratch (almost).

## Implementation layout

Although the original assignment suggested first implementing in the Cython
language, I decided to move to straight C so that I would have the time to both implement and debug as well as experiment with improvements to the vanilla network. My implementation uses the GNU Science Library (GSL) to perform matrix operations and in some cases directly calls on the seminal BLAS library. The code is broken up into three discrete modules as follows,

__network/network.h__

__mnist_network/mnist_network.h__


__mnist/mnist.h__

## Work log


> Network structure

> Feedforward




## Assignment

Build an NN with:

feedforward
backprop
stochastic gradient descent

Choose at least one of the tutorial datasets
Experiment with at least three of the following

Sigmoid vs ReLU
Cost functions
Number of nodes
Number of layers
Learning rate
Regularization parameter
Size of minibatch
Momentum
Adagrad, AdaDelta, RMSprop, ADAM
