#include "network.h"

int main() {
  network_t *net;
  gsl_matrix *a;
  int num_layers = 4;
  int layers[] = {10,30,30,10};
  printf("%s\n", "Initializing network");
  a = rand_gaussian_matrix(layers[0],1);
  printf("\n%s\n", "Network Overview");
  net = init_network(layers, num_layers, &sigmoid);
  assert(net->num_layers == num_layers);
  gsl_matrix_list_t *activations = gsl_matrix_list_malloc(num_layers);
  gsl_matrix_list_t *outputs = gsl_matrix_list_malloc(num_layers-1);
  gsl_matrix *m1 = feedforward(net, a, activations, outputs);
  printf("\n%s\n", "WEIGHTS");
  for (int i = 0; i < net->num_layers-1; i++) {
    gsl_matrix *m = net->weights[i];
    printf("layer %d, dim: %zu x %zu\n", i+1, m->size1, m->size2);
  }
  printf("\n%s\n", "BIASES");
  for (int i = 0; i < net->num_layers-1; i++) {
    gsl_matrix *m = net->biases[i];
    printf("layer %d, dim: %zu x %zu\n", i+1, m->size1, m->size2);
  }
  printf("\n%s\n", "ACTIVATIONS");
  for (int i = 0; i < activations->length; i++) {
    gsl_matrix *m1 = activations->data[i];
    printf("layer %d, dim: %zu x %zu\n", i, m1->size1, m1->size2);
  }
  printf("\n%s\n", "OUTPUTS");
  for (int j = 0; j < outputs->length; j++) {
    gsl_matrix *m2 = outputs->data[j];
    printf("layer %d, dim: %zu x %zu\n", j+1, m2->size1, m2->size2);
  }
  printf("\n%s\n", "FINAL OUTPUT");

  print_matrix (stdout, m1);

  printf("\n%s\n", "BACKPROP TEST");
  gsl_matrix *y;
  gsl_matrix_list_t *weight_grads;
  gsl_matrix_list_t *bias_grads;
  weight_grads = gsl_matrix_list_malloc(net->num_layers-1);
  bias_grads = gsl_matrix_list_malloc(net->num_layers-1);
  y = rand_gaussian_matrix(layers[0],1);
  backprop(net, a, y, weight_grads, bias_grads);

  printf("\n%s\n", "WEIGHT GRADS");
  for (int j = 0; j < weight_grads->length; j++) {
    gsl_matrix *m2 = weight_grads->data[j];
    printf("layer %d, dim: %zu x %zu\n", j+1, m2->size1, m2->size2);
  }

  printf("\n%s\n", "BIAS GRADS");
  for (int j = 0; j < bias_grads->length; j++) {
    gsl_matrix *m2 = bias_grads->data[j];
    printf("layer %d, dim: %zu x %zu\n", j+1, m2->size1, m2->size2);
  }








  // free shit
  gsl_matrix_free(a);
  free_network(net);

  return 0;
}
