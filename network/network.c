#include "network.h"

// BEGIN NETWORK FUNCTIONS

/*
  initialize a network
*/
network_t *init_network(int layers[], int num_layers, af_t *activation, cf_t *cost) {
  network_t *net = (network_t*) malloc(sizeof(network_t));
  net->num_layers = num_layers;
  memcpy(net->layers, layers, num_layers*sizeof(int));
  net->weights = (gsl_matrix**) malloc(sizeof(gsl_matrix*)*(num_layers-1));
  net->biases = (gsl_matrix**) malloc(sizeof(gsl_matrix*)*(num_layers-1));
  net->activation = activation;
  net->cost = cost;
  // Generate random biases and weights.
  for (int l = 1; l < num_layers; l++) {
    net->biases[l-1] = rand_gaussian_matrix(layers[l], 1);
    net->weights[l-1] = rand_gaussian_matrix(layers[l], layers[l-1]);
  }
  init_activations(net);
  init_outputs(net);
  net->bias_grads = init_bias_grads(net);
  net->weight_grads = init_weight_grads(net);
  net->delta_bias_grads = init_bias_grads(net);
  net->delta_weight_grads = init_weight_grads(net);

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
  gsl_matrix_list_free(net->activations);
  gsl_matrix_list_free(net->outputs);
  gsl_matrix_list_free(net->bias_grads);
  gsl_matrix_list_free(net->weight_grads);
  gsl_matrix_list_free(net->delta_weight_grads);
  gsl_matrix_list_free(net->delta_bias_grads);
  free(net->weights);
  free(net->biases);
  free(net->activation);
  free(net->cost);
  free(net);
}

/*
 perform feedforward proceedure on the network
 zs = outputs, as=activations
*/
void feedforward(network_t* net, gsl_matrix *a) {
  assert(net->layers[0] == a->size1);
  assert(net->activations->length == net->num_layers);
  assert(net->outputs->length == net->num_layers-1);
  gsl_matrix_memcpy(net->activations->data[0], a);

  for (int i = 0; i < (net->num_layers-1); i++) {
    activateLayer(net, i);
  }
}

// activateLayer is the inner loop of the feed forward
void activateLayer(network_t *net, int l) {
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, net->weights[l],
                net->activations->data[l], 0.0, net->outputs->data[l]);
  gsl_matrix_add(net->outputs->data[l], net->biases[l]);
  map_from(net->activation->f, net->activations->data[l+1], net->outputs->data[l]);
}

void backprop(network_t *net, gsl_matrix *target) {
  // dimensional check
  assert(net->delta_weight_grads->length == net->num_layers-1
          && net->delta_bias_grads->length == net->num_layers-1);

  // propogate forward thru the network
  gsl_matrix *delta;
  size_t asize = net->num_layers;
  size_t zsize = net->num_layers-1;
  size_t wgrad_size = net->delta_weight_grads->length;
  size_t bgrad_size = net->delta_bias_grads->length;

  gsl_matrix *delta_temp;

  // propogate backward thru the network
  gsl_matrix *cost_by_a = gsl_matrix_calloc(target->size1, target->size2);
  (*net->cost->f_p)(cost_by_a, net->activations->data[asize-1], target);
  gsl_matrix *sd_outputs = matrix_copy(net->outputs->data[zsize-1]);
  map(net->activation->f_p, sd_outputs);
  gsl_matrix_mul_elements(cost_by_a, sd_outputs);
  gsl_matrix_free(sd_outputs);
  delta = cost_by_a;
  delta_temp = gsl_matrix_alloc(net->weights[(wgrad_size-1)]->size2,
                                            delta->size2);

  // for (int i = 0; i < delta_bias_grads->length; i++) print_shape(delta_bias_grads->data[i], "bgrads");
  gsl_matrix_memcpy(net->delta_bias_grads->data[bgrad_size-1], delta);

  // gsl_blas_dgemm(f1, f2, alpha, A, B, beta, C)
  // C = alpha * f1(A) * f2(B) + beta * C
  // delta_weight_grads->data[wgrad_size-1] = gsl_matrix_alloc(delta->size1,
  //                                       (activations->data[asize-2])->size1);

  gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, delta,
                  net->activations->data[asize-2], 0.0, net->delta_weight_grads->data[wgrad_size-1]);

  for (int l = 2; l < net->num_layers; l++) {
    gsl_matrix *z = net->outputs->data[zsize-l];
    map(net->activation->f_p, z);
    gsl_matrix *sp = z;
    // gsl_matrix *delta_temp = gsl_matrix_alloc(net->weights[(wgrad_size-l+1)]->size2,
    //                                           delta->size2);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, net->weights[(wgrad_size-l+1)],
                    delta, 0.0, delta_temp);
    // gsl_matrix_free(delta);
    gsl_matrix_mul_elements(delta_temp, sp);
    delta = delta_temp;
    gsl_matrix_memcpy(net->delta_bias_grads->data[bgrad_size-l], delta);
    // delta_weight_grads->data[wgrad_size-l] = gsl_matrix_alloc(delta->size1,
    //                                       (activations->data[asize-l-1])->size1);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, delta,
                    net->activations->data[asize-l-1], 0.0, net->delta_weight_grads->data[wgrad_size-l]);

  }
  gsl_matrix_free(cost_by_a);
  gsl_matrix_free(delta);

  // free stuff
}

// void calculateErrorForLayer(network_t *net, int l) {
//
// }

// BEGIN MATRIX FUNCTIONS

void print_shape(gsl_matrix *m, const char *msg) {
  printf("%s: %zu x %zu\n", msg, m->size1, m->size2);
}


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
  gsl_rng_free(r);
  return m1;
}

/*
  make a new copy of a matrix
*/
gsl_matrix *matrix_copy(gsl_matrix *m) {
  gsl_matrix *new_m = gsl_matrix_calloc(m->size1, m->size2);
  gsl_matrix_memcpy(new_m, m);
  return new_m;
}

gsl_matrix_list_t *init_bias_grads(network_t *net) {
  gsl_matrix_list_t *ml = (gsl_matrix_list_t*) malloc(sizeof(gsl_matrix_list_t));
  ml->length = net->num_layers-1;
  ml->data = (gsl_matrix**) malloc(sizeof(gsl_matrix*)*(net->num_layers-1));
  for (int l = 1; l < net->num_layers; l++) {
    ml->data[l-1] = gsl_matrix_calloc(net->layers[l], 1);
  }
  return ml;
}


gsl_matrix_list_t *init_weight_grads(network_t *net) {
  gsl_matrix_list_t *ml = (gsl_matrix_list_t*) malloc(sizeof(gsl_matrix_list_t));
  ml->length = net->num_layers-1;
  ml->data = (gsl_matrix**) malloc(sizeof(gsl_matrix*)*(net->num_layers-1));
  for (int l = 1; l < net->num_layers; l++) {
    ml->data[l-1] = gsl_matrix_calloc(net->layers[l], net->layers[l-1]);
  }
  return ml;
}

void init_outputs(network_t *net) {
  gsl_matrix_list_t *ml = (gsl_matrix_list_t*) malloc(sizeof(gsl_matrix_list_t));
  ml->length = net->num_layers-1;
  ml->data = (gsl_matrix**) malloc(sizeof(gsl_matrix*)*(net->num_layers-1));
  for (int l = 1; l < net->num_layers; l++) {
    ml->data[l-1] = gsl_matrix_calloc(net->layers[l], 1);
  }
  net->outputs = ml;
}

void init_activations(network_t *net) {
  gsl_matrix_list_t *ml = (gsl_matrix_list_t*) malloc(sizeof(gsl_matrix_list_t));
  ml->length = net->num_layers;
  ml->data = (gsl_matrix**) malloc(sizeof(gsl_matrix*)*(net->num_layers));
  for (int l = 0; l < net->num_layers; l++) {
    ml->data[l] = gsl_matrix_calloc(net->layers[l], 1);
  }
  net->activations = ml;
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

void gsl_matrix_list_free(gsl_matrix_list_t *ml) {
  for (int i = 0; i < ml->length; i++) {
    gsl_matrix_free(ml->data[i]);
  }
  free(ml->data);
  free(ml);
}

void gsl_matrix_list_free_matrices(gsl_matrix_list_t *ml) {
  for (int i = 0; i < ml->length; i++) {
    gsl_matrix_free(ml->data[i]);
  }
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

/*
  map a function f: double -> double from src onto dest
*/
void map_from(double (*f)(double), gsl_matrix *dest, gsl_matrix *src) {
  double x = 0;
  assert(same_shape(dest, src));
  for (size_t i = 0; i < src->size1; i++) {
    for (size_t j = 0; j < src->size2; j++) {
      x = gsl_matrix_get(src, i, j);
      gsl_matrix_set(dest, i, j, ((*f)(x)));
    }
  }
}

// BEGIN AUXILIARY FUNCTIONS


void save(network_t *net) {
  char filepath[BUFFER_SIZE];
  char buff[BUFFER_SIZE];
  time_t now = time(NULL);
  strftime(buff, 20, "%Y-%m-%d%H:%M:%S", localtime(&now));
  sprintf(filepath, "mnist_network%s.res", buff);
  FILE *f = fopen(filepath, "w");
  if (f == NULL) {
      printf("Error opening file!\n");
      exit(1);
  }
  for (int i = 0; i < net->num_layers-1; i++) print_matrix(f, net->weights[i]);
  fprintf(f, "\n");
  for (int i = 0; i < net->num_layers-1; i++) print_matrix(f, net->biases[i]);
  fprintf(f, "\n");
  fclose(f);
}



af_t *use_sigmoid() {
  af_t *a = (af_t*)malloc(sizeof(af_t));
  a->f = &sigmoid;
  a->f_p = &sigmoid_prime;
  return a;
}

double sigmoid_prime(double x) {
  return (sigmoid(x) * (1.0 - sigmoid(x)));
}

double sigmoid(double x) {
  return (1.0 / (1.0 + exp(-(1.0) * x)));
}

double relu(double x) {
  return (x > 0.0) ? x : 0;
}



cf_t *use_quad_cost() {
  cf_t *c = (cf_t*)malloc(sizeof(cf_t));
  // c->f = &quad_cost;
  c->f_p = &quad_cost_p;
  return c;
}

// quad_cost_p applies the derivative of the quadratic cost function
void quad_cost_p(gsl_matrix *dest, gsl_matrix *a, gsl_matrix *y) {
  // calculate a - y
  // assert(dest != NULL && a != NULL && y != NULL);
  // assert(same_shape(a, y) && same_shape(y, dest));
  gsl_matrix_add(dest, y);  // dest = y
  gsl_matrix_scale(dest, -1.0); // dest = - y
  gsl_matrix_add(dest, a);  // dest = a + (-y) = a - y
}

// gsl_matrix *quad_cost(gsl_matrix *final_activation,
//                                  gsl_matrix *targets) {
//   // a_L - y
//   assert(same_shape(final_activation, targets));
//   gsl_matrix_sub(final_activation, targets);
//   gsl_matrix_memcpy(m, final_activation);
//   return m;
// }
