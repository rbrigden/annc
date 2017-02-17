#ifndef __NETWORK_H__
#define  __NETWORK_H__

#include "../csapp.h"
#include <assert.h>
#include <stdbool.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_randist.h>
#include <time.h>


#define BUFFER_SIZE 256
// layers in the network
#define MAX_LAYERS 64
// standard deviation of the gaussian distribution
#define SIGMA 1

typedef struct gsl_matrix_list {
  int length;
  gsl_matrix **data;
} gsl_matrix_list_t;

typedef struct network {
  int num_layers;
  double (*activation)(double);
  int layers[MAX_LAYERS];
  gsl_matrix **weights;
  gsl_matrix **biases;
  gsl_matrix_list_t *activations;
  gsl_matrix_list_t *outputs;
} network_t;

// network functions
network_t *init_network(int layers[], int num_layers,
                                      double (*activation)(double));
void free_network(network_t *net);
void feedforward(network_t* net, gsl_matrix *a);
void activateLayer(network_t *net, int l);

void backprop(network_t *net, gsl_matrix *input, gsl_matrix *target,
              gsl_matrix_list_t *weight_grads, gsl_matrix_list_t *bias_grads);
// auxiliary functions
double sigmoid(double z);
double sigmoid_prime(double x);
double relu(double z);
gsl_matrix *quad_cost_derivative(gsl_matrix *final_activation,
                                 gsl_matrix *targets);
void save(network_t *net);


// matrix functions
gsl_matrix *rand_gaussian_matrix(size_t rows, size_t cols);
void map(double (*f)(double), gsl_matrix *m);
void map_from(double (*f)(double), gsl_matrix *dest, gsl_matrix *src);
void print_matrix(FILE *f, const gsl_matrix *m);
bool same_shape(gsl_matrix *a, gsl_matrix *b);
gsl_matrix_list_t *gsl_matrix_list_malloc(size_t length);
gsl_matrix *matrix_copy(gsl_matrix *m);
void gsl_matrix_list_free(gsl_matrix_list_t *ml); // the whole shebang
void gsl_matrix_list_free_matrices(gsl_matrix_list_t *ml); // just the matrices
gsl_matrix_list_t *init_bias_grads(network_t *net);
gsl_matrix_list_t *init_weight_grads(network_t *net);
void print_shape(gsl_matrix *m, const char *msg);
void init_outputs(network_t *net);
void init_activations(network_t *net);

#endif
