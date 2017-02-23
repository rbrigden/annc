#ifndef __TRAINING_H__
#define  __TRAINING_H__

#include "../mnist/mnist.h"
#include "../network/network.h"

#define MU 0.9
#define LAMBDA 0.8

void stochastic_gradient_descent(network_t *net, set_loader_t *train_loader,
      set_loader_t *test_loader, int mini_batch_size, int epochs, double eta);
gsl_matrix *image_to_matrix(image_t *img, size_t width, size_t height);
gsl_matrix *mnist_target_matrix(image_t *img);
void update_mini_batch(network_t *net, set_loader_t *loader,
                  gsl_matrix_list_t *vw, gsl_matrix_list_t *vb, int mini_batch_size, double eta);
int evaluate(network_t *net, set_loader_t *test_loader);
gsl_matrix *image_to_matrix(image_t *img, size_t width, size_t height);
gsl_matrix *mnist_target_matrix(image_t *img);


#endif
