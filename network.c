#include "network.h"


/*
  initialize a network
*/
network_t *init_network(int layers[], int num_layers) {
  network_t *net = (network_t*) malloc(sizeof(network_t));
  net->num_layers = num_layers;
  memcpy(net->layers, layers, num_layers*sizeof(int));
  net->weights = (gsl_matrix**) malloc(sizeof(gsl_matrix*)*(num_layers-1));
  net->biases = (gsl_matrix**) malloc(sizeof(gsl_matrix*)*(num_layers-1));

  // Generate random biases and weights.
  for (int l = 1; l < num_layers; l++) {
    net->biases[l-1] = rand_gaussian_matrix(layers[l], 1);
    net->weights[l-1] = rand_gaussian_matrix(layers[l], layers[l-1]);
  }

  return net;
}

/*
 free a network
*/
void free_network(network_t* net) {
  for (int i = 0; i < net->num_layers-1; i++) {
    gsl_matrix_free(net->weights[i]);
    gsl_matrix_free(net->biases[i]);
  }
  free(net->weights);
  free(net->biases);
  free(net);
}

/*
 print a gsl_matrix
*/
int print_matrix(FILE *f, const gsl_matrix *m) {
  int status, n = 0;
  for (size_t i = 0; i < m->size1; i++) {
  for (size_t j = 0; j < m->size2; j++) {
  if ((status = fprintf(f, "%g ", gsl_matrix_get(m, i, j))) < 0)
  return -1;
                          n += status;
                  }
  if ((status = fprintf(f, "\n")) < 0)
  return -1;
                  n += status;
          }
  return n;
}

/*
 initialize random gaussian matrix with standard deviation SIGMA
*/
gsl_matrix *rand_gaussian_matrix(size_t rows, size_t cols) {
  gsl_matrix *m1;
  const gsl_rng_type * T;
  gsl_rng * r;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc(T);

  m1 = gsl_matrix_calloc (rows, cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
       gsl_matrix_set (m1, i, j, gsl_ran_gaussian(r, SIGMA));
    }
  }
  return m1;
}
