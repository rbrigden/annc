#include "network.h"



int main() {
  network_t *net;
  gsl_matrix *a;
  int num_layers = 4;
  int layers[] = {10,30,30,10};
  printf("%s\n", "Initializing network");
  a = rand_gaussian_matrix(10,1);
  net = init_network(layers, num_layers, &sigmoid);
  assert(net->num_layers == num_layers);
  print_matrix (stdout, feedforward(net, a));

  // free shit
  gsl_matrix_free(a);
  free_network(net);

  return 0;
}
