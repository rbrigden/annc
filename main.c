#include "mnist_network/mnist_network.h"

#define EPOCHS 30
#define ETA 1.0
#define MINI_BATCH_SIZE 50
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
  net = init_network(layers, num_layers, &sigmoid);

  if (verify_data()) {
    printf("%s\n", "woohoo, verified");
  } else {
    printf("%s\n", "whoops, not verified");
    return 1;
  }
  train_set = init_set_loader(TRAIN_IMAGES, TRAIN_LABELS);
  test_set = init_set_loader(TEST_IMAGES, TEST_LABELS);
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
    net = init_network(layers, num_layers, &sigmoid);

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

  y = rand_gaussian_matrix(layers[num_layers-1],1);


  backprop(net, a, y, weight_grads, bias_grads);

  printf("\n%s\n", "WEIGHT GRADS");
  for (int j = 0; j < weight_grads->length; j++) {
    gsl_matrix *m2 = weight_grads->data[j];
    printf("layer %d, dim: %zu x %zu\n", j+1, m2->size1, m2->size2);
    // print_matrix(stdout, m2);
  }

  printf("\n%s\n", "BIAS GRADS");
  for (int j = 0; j < bias_grads->length; j++) {
    gsl_matrix *m2 = bias_grads->data[j];
    // print_matrix(stdout, m2);
    printf("layer %d, dim: %zu x %zu\n", j+1, m2->size1, m2->size2);
  }

  // free shit
  gsl_matrix_free(a);
  gsl_matrix_free(y);
  gsl_matrix_list_free(weight_grads);
  gsl_matrix_list_free(bias_grads);
  gsl_matrix_list_free(activations);
  gsl_matrix_list_free(outputs);
  gsl_matrix_free(m1);

  free_network(net);

  return 0;
}
