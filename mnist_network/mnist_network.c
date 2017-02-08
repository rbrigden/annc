#include "mnist_network.h"



void stochastic_gradient_descent(network_t *net, set_loader_t *train_loader,
      set_loader_t *test_loader, int mini_batch_size, int epochs, double eta) {
  int mini_batches = (train_loader->total/mini_batch_size);
  for (size_t e = 0; e < epochs; e++) {
    shuffle(train_loader);
    for (int m = 0; m < mini_batches; m++) {
      // printf("Batch %d\n", m+1);
      update_mini_batch(net, train_loader, mini_batch_size, eta);
    }
    printf("\nEpoch: %zu, accuracy %4f\n", e, evaluate(net, test_loader));
    shuffle(test_loader);
  }
}

gsl_matrix *image_to_matrix(image_t *img, size_t width, size_t height) {
  gsl_matrix *image_matrix;
  size_t len = width * height;
  image_matrix = gsl_matrix_alloc(len, 1);
  for (size_t i = 0; i < len; i++) {
    gsl_matrix_set (image_matrix, i, 0, (double)(img->data[i]));
  }
  return image_matrix;
}

gsl_matrix *mnist_target_matrix(image_t *img) {
  gsl_matrix *target_matrix;
  target_matrix = gsl_matrix_calloc(10, 1);
  gsl_matrix_set (target_matrix, (size_t)img->label, 0, 1);
  return target_matrix;
}

void update_mini_batch(network_t *net, set_loader_t *loader,
                                int mini_batch_size, double eta) {

  gsl_matrix_list_t *weight_grads, *delta_weight_grads;
  gsl_matrix_list_t *bias_grads, *delta_bias_grads;
  gsl_matrix *input;
  gsl_matrix *target;
  image_t *img;

  // initialize local gradients

  delta_weight_grads = init_weight_grads(net);
  delta_bias_grads = init_bias_grads(net);
  // printf("%zu\n", delta_bias_grads->data[delta_bias_grads->length-1]->size1);
  weight_grads = init_weight_grads(net);
  bias_grads = init_bias_grads(net);

  for (int i = 0; i < mini_batch_size; i++) {
    img = get_next_image(loader);
    input = image_to_matrix(img, loader->height, loader->width);
    target = mnist_target_matrix(img);
    backprop(net, input, target, delta_weight_grads, delta_bias_grads);
    gsl_matrix_free(input);
    gsl_matrix_free(target);
    for (int l = 0; l < net->num_layers-1; l++) {
      gsl_matrix_add(weight_grads->data[l], delta_weight_grads->data[l]);
      gsl_matrix_add(bias_grads->data[l], delta_bias_grads->data[l]);
    }
    for (int l = 0; l < net->num_layers-1; l++) {
      // w -= eta/len(mini_batch) * delta_weight_grad
      double scaler = eta / ((double)mini_batch_size);
      gsl_matrix_scale(delta_weight_grads->data[l], scaler);
      gsl_matrix_sub(net->weights[l], delta_weight_grads->data[l]);

      gsl_matrix_scale(delta_bias_grads->data[l], scaler);
      gsl_matrix_sub(net->biases[l], delta_bias_grads->data[l]);
    }
  }

  gsl_matrix_list_free(delta_weight_grads);
  gsl_matrix_list_free(delta_bias_grads);

  gsl_matrix_list_free(weight_grads);
  gsl_matrix_list_free(bias_grads);
}

double evaluate(network_t *net, set_loader_t *test_loader) {
  gsl_matrix *input;
  gsl_matrix *final_activation;
  gsl_matrix_list_t *activations;
  gsl_matrix_list_t *outputs;
  size_t imax, jmax;
  image_t *img;
  double sum = 0;
  for (size_t m = 0; m < test_loader->total; m++) {
    img = get_next_image(test_loader);
    input = image_to_matrix(img, test_loader->height, test_loader->width);
    activations = gsl_matrix_list_malloc(net->num_layers);
    outputs = gsl_matrix_list_malloc(net->num_layers-1);
    final_activation = feedforward(net, input, activations, outputs);
    gsl_matrix_max_index(final_activation, &imax, &jmax);
    sum += (imax == (size_t)img->label) ? 1 : 0;
    gsl_matrix_list_free(activations);
    gsl_matrix_list_free(outputs);
  }
  return (sum / ((double)test_loader->total));
}
