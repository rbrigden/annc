#include "network/network.h"


static void test_init_network() {
  network_t *net;
  int num_layers = 4;
  int layers[] = {1,10,10,1};
  net = init_network(layers, num_layers, &sigmoid);
  assert(net->num_layers == num_layers);

  // test matrix dimensions
  for (int i = 0; i < num_layers-1; i++) {
    gsl_matrix *m1 = net->weights[i];
    gsl_matrix *m2 = net->biases[i];

    // check weight dimensions
    assert(m1->size1 == layers[i+1]);
    assert(m1->size2 == layers[i]);

    // check biases dimensions
    assert(m2->size1 == layers[i+1]);
    assert(m2->size2 == 1);
  }

  free_network(net);
}

static void test_map() {
  gsl_matrix *m1;
  gsl_matrix *m2;
  m1 = gsl_matrix_calloc(10, 10);
  m2 = gsl_matrix_alloc(10, 10);
  gsl_matrix_set_all (m2, 2);
  double add2 (double x) { return x + 2; }
  map(&add2, m1);
  assert(gsl_matrix_equal(m1,m2));
}


int main() {
  printf("%s\n", "Testing init_network");
  test_init_network();
  printf("%s\n", "Testing test_map");
  test_map();
  printf("%s\n", "All tests passed.");
  return 0;
}
