#include "network.h"
#include <assert.h>


static void test_init_network() {
  network_t *net;
  int num_layers = 4
  int layers[num_layers] = {1,10,10,1}
  net = init_network(layers, num_layers);
  assert(net->num_layers == num_layers);
}


int main() {
  test_init_network();
  return 0;
}
