#ifndef __MNIST_LOADER_H_
#define  __MNIST_LOADER_H_

#include "../mnist/mnist_loader.h"
#include "../network/network.h"


void stochastic_gradient_descent(network_t *net, set_loader_t *train_loader,
      set_loader_t *test_loader, int mini_batch_size, int epochs, double eta);
gsl_matrix *image_to_matrix(image_t *img, size_t width, size_t height);
gsl_matrix *mnist_target_matrix(image_t *img);
void update_mini_batch(network_t *net, set_loader_t *loader,
                                int mini_batch_size, double eta);
double evaluate(network_t *net, set_loader_t *test_loader);
gsl_matrix *image_to_matrix(image_t *img, size_t width, size_t height);
gsl_matrix *mnist_target_matrix(image_t *img);





#endif
