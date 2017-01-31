#ifndef __NETWORK_H__
#define  __NETWORK_H__

#include "csapp.h"
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
  int layers[MAX_LAYERS];
  gsl_matrix **weights;
  gsl_matrix **biases;
} network_t;

// network functions
network_t *init_network(int layers[], int num_layers);
void free_network(network_t *net);

// matrix functions
gsl_matrix *rand_gaussian_matrix(size_t rows, size_t cols);
int print_matrix(FILE *f, const gsl_matrix *m);



#endif
