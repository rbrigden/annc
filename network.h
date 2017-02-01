#ifndef __NETWORK_H__
#define  __NETWORK_H__

#include "csapp.h"
#include <assert.h>
#include <stdbool.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_randist.h>



#define MAX_OBJECT_SIZE 102400
// layers in the network
#define MAX_LAYERS 64
// standard deviation of the gaussian distribution
#define SIGMA 1


typedef struct network {
  int num_layers;
  double (*activation)(double);
  int layers[MAX_LAYERS];
  gsl_matrix **weights;
  gsl_matrix **biases;
} network_t;

// network functions
network_t *init_network(int layers[], int num_layers,
                                      double (*activation)(double));
void free_network(network_t *net);
gsl_matrix *feedforward(network_t* net, gsl_matrix *a);


// activation functions
double sigmoid(double z);
double relu(double z);


// matrix functions
gsl_matrix *rand_gaussian_matrix(size_t rows, size_t cols);
void map(double (*f)(double), gsl_matrix *m);
void print_matrix(FILE *f, const gsl_matrix *m);
bool same_shape(gsl_matrix *a, gsl_matrix *b);



#endif
