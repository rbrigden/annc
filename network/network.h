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

typedef struct af {
  double (*f)(double); // activation function
  double (*f_p)(double); // activation function derivative
} af_t;

typedef struct cf {
  double (*f)(gsl_matrix*, gsl_matrix*); // cost function
  void (*f_p)(gsl_matrix*, gsl_matrix*, gsl_matrix*); // cost function prime
} cf_t;

typedef struct network {
  af_t *activation;
  cf_t *cost;

  int num_layers;
  int layers[MAX_LAYERS];
  gsl_matrix **weights;
  gsl_matrix **biases;
  gsl_matrix_list_t *activations;
  gsl_matrix_list_t *outputs;
  gsl_matrix_list_t *weight_grads;
  gsl_matrix_list_t *bias_grads;
  gsl_matrix_list_t *delta_weight_grads;
  gsl_matrix_list_t *delta_bias_grads;
} network_t;

// network functions
network_t *init_network(int layers[], int num_layers, af_t *activation, cf_t *cost);
void free_network(network_t *net);
void feedforward(network_t* net, gsl_matrix *a);
void activateLayer(network_t *net, int l);

void backprop(network_t *net, gsl_matrix *target);

// activation functions
af_t *use_sigmoid();
double sigmoid(double z);
double sigmoid_prime(double x);
double relu(double z);

// cost functions
cf_t *use_quad_cost();
void quad_cost_p(gsl_matrix *dest, gsl_matrix *a, gsl_matrix *y);
double quad_cost(gsl_matrix *a, gsl_matrix *y);

// auxiliary functions
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
void gsl_matrix_list_set_zero(gsl_matrix_list_t *ml);

#endif
