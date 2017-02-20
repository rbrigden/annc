#include "mnist_network/mnist_network.h"

#define EPOCHS 30
#define ETA 1.0
#define MINI_BATCH_SIZE 1000
#define LAYERS {(28*28), 30, 30, 10}
#define NUM_LAYERS 4

static int net_example();
static int train_mnist();
static int mnist_example_load();

int main() {
  train_mnist();
  return 0;
}

int train_mnist() {
  set_loader_t *train_set;
  set_loader_t *test_set;
  network_t *net;
  int num_layers = NUM_LAYERS;
  int layers[] = LAYERS;
  printf("%s\n", "Initializing network");
  // cf_t *cost = use_quad_cost();
  cf_t *cost = use_cross_entropy_cost();

  af_t *activation = use_sigmoid();
  net = init_network(layers, num_layers, activation, cost);

  if (verify_data()) {
    printf("%s\n", "woohoo, verified");
  } else {
    printf("%s\n", "whoops, not verified");
    return 1;
  }
  train_set = init_set_loader(TRAIN_IMAGES, TRAIN_LABELS);
  test_set = init_set_loader(TEST_IMAGES, TEST_LABELS);
  // test_set = init_set_loader(TRAIN_IMAGES, TRAIN_LABELS);
  // train_set->total = 100;
  // test_set->total = 100;

  printf("\nEpochs: %d, Eta: %4f, MBS: %d\n\n", EPOCHS, ETA, MINI_BATCH_SIZE);

  stochastic_gradient_descent(net, train_set, test_set, MINI_BATCH_SIZE, EPOCHS, ETA);
  set_loader_free(train_set);
  set_loader_free(test_set);
  free_network(net);
  return 0;
}

int mnist_example_load() {
    set_loader_t *train_set;
    set_loader_t *test_set;
    network_t *net;
    int num_layers = 3;
    int layers[] = {(28*28),30,10};
    printf("%s\n", "Initializing network");
    cf_t *cost = use_quad_cost();
    af_t *activation = use_sigmoid();
    net = init_network(layers, num_layers, activation, cost);

    if (verify_data()) {
      printf("%s\n", "woohoo, verified");
    } else {
      printf("%s\n", "whoops, not verified");
      return 1;
    }
    train_set = init_set_loader(TRAIN_IMAGES, TRAIN_LABELS);
    test_set = init_set_loader(TEST_IMAGES, TEST_LABELS);
    printf("%s\n", "No shuffle");
    for (int i = 0; i < 5; i++) image_print(get_next_image(train_set), train_set->height);
    shuffle(train_set);
    printf("%s\n", "Post shuffle");
    for (int i = 0; i < 5; i++) image_print(get_next_image(train_set), train_set->height);

    set_loader_free(train_set);
    set_loader_free(test_set);
    free_network(net);
    return 0;
}


int net_example() {
  network_t *net;
  gsl_matrix *a;
  int num_layers = 4;
  int layers[] = {10,30,30,10};
  printf("%s\n", "Initializing network");
  a = rand_gaussian_matrix(layers[0],1);
  printf("\n%s\n", "Network Overview");
  net = init_network(layers, num_layers, use_sigmoid(), use_quad_cost());
  assert(net->num_layers == num_layers);
  feedforward(net, a);


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
  for (int i = 0; i < net->activations->length; i++) {
    gsl_matrix *m1 = net->activations->data[i];
    printf("layer %d, dim: %zu x %zu\n", i, m1->size1, m1->size2);
  }
  printf("\n%s\n", "OUTPUTS");
  for (int j = 0; j < net->outputs->length; j++) {
    gsl_matrix *m2 = net->outputs->data[j];
    printf("layer %d, dim: %zu x %zu\n", j+1, m2->size1, m2->size2);
  }
  printf("\n%s\n", "FINAL OUTPUT");

  print_matrix (stdout, net->activations->data[net->num_layers-1]);

  printf("\n%s\n", "BACKPROP TEST");
  gsl_matrix *y;

  y = rand_gaussian_matrix(layers[num_layers-1],1);

  feedforward(net, a);
  backprop(net, y);

  printf("\n%s\n", "WEIGHT GRADS");
  for (int j = 0; j < net->weight_grads->length; j++) {
    gsl_matrix *m2 = net->weight_grads->data[j];
    printf("layer %d, dim: %zu x %zu\n", j+1, m2->size1, m2->size2);
    // print_matrix(stdout, m2);
  }

  printf("\n%s\n", "BIAS GRADS");
  for (int j = 0; j < net->bias_grads->length; j++) {
    gsl_matrix *m2 = net->bias_grads->data[j];
    // print_matrix(stdout, m2);
    printf("layer %d, dim: %zu x %zu\n", j+1, m2->size1, m2->size2);
  }

  // free shit
  gsl_matrix_free(a);
  gsl_matrix_free(y);
  free_network(net);

  return 0;
}
