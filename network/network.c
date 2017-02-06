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
 zs = outputs, as=activations
*/
gsl_matrix *feedforward(network_t* net, gsl_matrix *a,
                    gsl_matrix_list_t *activations, gsl_matrix_list_t *outputs) {
  assert(net->layers[0] == a->size1);
  assert(activations->length == net->num_layers);
  assert(outputs->length == net->num_layers-1);
  gsl_matrix *tempa;  // temp activations
  gsl_matrix *w;  // weights
  gsl_matrix *b;  // biases
  activations->data[0] = a; // add the first activation
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
    outputs->data[i] = gsl_matrix_alloc(tempa->size1, tempa->size2);
    gsl_matrix_memcpy(outputs->data[i], tempa);
    map(net->activation, tempa);
    a = tempa;
    activations->data[i+1] = gsl_matrix_alloc(a->size1, a->size2);
    gsl_matrix_memcpy(activations->data[i+1], a);
  }
  return a;
}

void backprop(network_t *net, gsl_matrix *input, gsl_matrix *target,
              gsl_matrix_list_t *weight_grads, gsl_matrix_list_t *bias_grads) {
  // dimensional check
  assert(weight_grads->length == net->num_layers-1
          && bias_grads->length == net->num_layers-1);

  // propogate forward thru the network
  gsl_matrix *delta;
  size_t asize = net->num_layers;
  size_t zsize = net->num_layers-1;
  size_t wgrad_size = weight_grads->length;
  size_t bgrad_size = bias_grads->length;
  gsl_matrix_list_t *activations = gsl_matrix_list_malloc(asize);
  gsl_matrix_list_t *outputs = gsl_matrix_list_malloc(zsize);
  gsl_matrix *final_activation = feedforward(net, input, activations, outputs);

  // propogate backward thru the network
  gsl_matrix *cost_by_a = quad_cost_derivative(final_activation, target);
  gsl_matrix *sd_outputs = matrix_copy(outputs->data[zsize-1]);
  map(&sigmoid_prime, sd_outputs);
  gsl_matrix_mul_elements(cost_by_a, sd_outputs);
  delta = cost_by_a;
  bias_grads->data[bgrad_size-1] = matrix_copy(delta);

  // gsl_blas_dgemm(f1, f2, alpha, A, B, beta, C)
  // C = alpha * f1(A) * f2(B) + beta * C
  weight_grads->data[wgrad_size-1] = gsl_matrix_alloc(delta->size1,
                                        (activations->data[asize-2])->size1);
  gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, delta,
                  activations->data[asize-2], 0.0, weight_grads->data[wgrad_size-1]);

  for (int l = 2; l < net->num_layers; l++) {
    gsl_matrix *z = outputs->data[zsize-l];
    gsl_matrix *sp = matrix_copy(z);
    map(&sigmoid_prime, sp);
    gsl_matrix *delta_temp = gsl_matrix_alloc(net->weights[(wgrad_size-l+1)]->size2,
                                              delta->size2);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, net->weights[(wgrad_size-l+1)],
                    delta, 0.0, delta_temp);
    gsl_matrix_mul_elements(delta_temp, sp);
    delta = delta_temp;
    bias_grads->data[bgrad_size-l] = matrix_copy(delta);
    weight_grads->data[wgrad_size-l] = gsl_matrix_alloc(delta->size1,
                                          (activations->data[asize-l-1])->size1);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, delta,
                    activations->data[asize-l-1], 0.0, weight_grads->data[wgrad_size-l]);
  }

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
  for (size_t i = 0; i < m->size1; i++) {
    for (size_t j = 0; j < m->size2; j++) {
            fprintf(f, "%g ", gsl_matrix_get(m, i, j));
    }
    fprintf(f, "\n");
  }
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
  make a new copy of a matrix
*/
gsl_matrix *matrix_copy(gsl_matrix *m) {
  gsl_matrix *new_m = gsl_matrix_alloc(m->size1, m->size2);
  gsl_matrix_memcpy(new_m, m);
  return new_m;
}

/*
  create an array of gsl_matrices
*/
gsl_matrix_list_t *gsl_matrix_list_malloc(size_t length) {
  gsl_matrix_list_t *ml = (gsl_matrix_list_t*) malloc(sizeof(gsl_matrix_list_t));
  ml->length = length;
  ml->data = (gsl_matrix**) malloc(sizeof(gsl_matrix*)*(length));
  return ml;
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

gsl_matrix *quad_cost_derivative(gsl_matrix *final_activation,
                                 gsl_matrix *targets) {
  // a_L - y
  assert(same_shape(final_activation, targets));
  gsl_matrix *m = gsl_matrix_alloc(targets->size1, targets->size2);
  gsl_matrix_sub(final_activation, targets);
  gsl_matrix_memcpy(m, final_activation);
  return m;
}

double sigmoid_prime(double x) {
  return (sigmoid(x) * (1 - sigmoid(x)));
}

double sigmoid(double x) {
  return (1 / (1 + exp(-1 * x)));
}

double relu(double x) {
  return (x > 0.0) ? x : 0;
}
