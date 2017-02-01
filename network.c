#include "network.h"



// BEGIN NETWORK FUNCTIONS

/*
  initialize a network
*/
network_t *init_network(int layers[], int num_layers,
                                      double (*activation)(double)) {
  network_t *net = (network_t*) malloc(sizeof(network_t));
  net->num_layers = num_layers;
  memcpy(net->layers, layers, num_layers*sizeof(int));
  net->weights = (gsl_matrix**) malloc(sizeof(gsl_matrix*)*(num_layers-1));
  net->biases = (gsl_matrix**) malloc(sizeof(gsl_matrix*)*(num_layers-1));
  net->activation = activation;
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
 perform feedforward proceedure on the network
*/
gsl_matrix *feedforward(network_t* net, gsl_matrix *a) {
  assert(net->layers[0] == a->size1);
  gsl_matrix *tempa; // temp activations
  gsl_matrix *w; // weights
  gsl_matrix *b; // biases
  for (int i = 0; i < (net->num_layers-1); i++) {
    w = net->weights[i];
    b = net->biases[i];
    tempa = gsl_matrix_alloc(w->size1, a->size2);
    // gsl_blas_dgemm(f1, f2, alpha, A, B, beta, C)
    // C = alpha * f1(A) * f2(B) + beta * C
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
                                1.0, w, a,
                                0.0, tempa);
    assert(same_shape(tempa, b));
    gsl_matrix_add(tempa, b);
    map(net->activation, tempa);
    a = tempa;
  }
  return a;
}





// BEGIN MATRIX FUNCTIONS

/*
  check if two gsl_matrix have the same shape
*/
bool same_shape(gsl_matrix *a, gsl_matrix *b) {
  return (a->size1 == b->size1 && a->size2 == b->size2);
}

/*
 print a gsl_matrix
*/
void print_matrix(FILE *f, const gsl_matrix *m) {
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

/*
  map a function f: double -> double onto a gsl_matrix
*/
void map(double (*f)(double), gsl_matrix *m) {
  double x = 0;
  for (size_t i = 0; i < m->size1; i++) {
    for (size_t j = 0; j < m->size2; j++) {
      x = gsl_matrix_get(m, i, j);
      gsl_matrix_set(m, i, j, ((*f)(x)));
    }
  }
}

// BEGIN AUXILIARY FUNCTIONS

double sigmoid(double x) {
  return (1 / (1 + exp(-1 * x)));
}

double relu(double x) {
  return (x > 0.0) ? x : 0;
}
