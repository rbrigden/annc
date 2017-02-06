#include "mnist_loader.h"
#include <assert.h>

/*
  verify_data checks that the relevant data files are available.
*/
bool verify_data() {
  int train_images_fd;
  int train_labels_fd;
  int test_images_fd;
  int test_labels_fd;

  int magic_number = 0;

  // TRAINING SET IMAGE FILE
  train_images_fd = open(TRAIN_IMAGES, O_RDONLY, 0);
  read(train_images_fd, &magic_number, 4);
  close(train_images_fd);
  if (TRAIN_IMAGES_MN != (htonl(magic_number))) {
    return false;
  }

  // TRAINING LABELS  FILE
  train_labels_fd = open(TRAIN_LABELS, O_RDONLY, 0);
  read(train_labels_fd, &magic_number, 4);
  close(train_labels_fd);
  if (TRAIN_LABELS_MN != (htonl(magic_number))) {
    return false;
  }

  // TEST IMAGES FILE
  test_images_fd = open(TEST_IMAGES, O_RDONLY, 0);
  read(test_images_fd, &magic_number, 4);
  close(test_images_fd);
  if (TEST_IMAGES_MN != (htonl(magic_number))) {
    return false;
  }

  // TEST LABELS FILE
  test_labels_fd = open(TEST_LABELS, O_RDONLY, 0);
  read(test_labels_fd, &magic_number, 4);
  close(test_labels_fd);
  if (TEST_LABELS_MN != (htonl(magic_number))) {
    return false;
  }

  return true;
}

/*
  init_set_loader initializes the set loader with the data and respective labels
*/
set_loader_t *init_set_loader(const char *data_file, const char *label_file) {
  int images_fd;
  int labels_fd;
  int num_images;
  int num_labels;
  int width, height;
  size_t image_data_size;
  uint8_t *image_data;
  set_loader_t *set;
  image_t *image;


  printf("%s\n", "Initializing set loader");

  // READ DATA HEADER
  printf("\n%s\n", "Reading data header");
  images_fd = open(data_file, O_RDONLY, 0);
  lseek(images_fd, 4, SEEK_SET); // read past magic number
  read(images_fd, &num_images, 4); // read sizes
  read(images_fd, &height, 4);
  read(images_fd, &width, 4);
  // fix endian-ness
  num_images = htonl(num_images);
  width = htonl(width);
  height = htonl(height);
  printf("Number of images: %d\n", num_images);
  printf("Image height: %d\n", height);
  printf("Image width: %d\n", width);

  // READ LABEL HEADER
  printf("\n%s\n", "Reading label header");
  labels_fd = open(label_file, O_RDONLY, 0);
  lseek(labels_fd, 4, SEEK_SET);
  read(labels_fd, &num_labels, 4); // read labels
  // fix endian-ness
  num_labels = htonl(num_labels);
  printf("Number of labels: %d\n", num_labels);
  assert(num_labels == num_images);

  // READ IMAGE DATA
  image_data_size = width * height * num_images * sizeof(uint8_t);
  image_data = (uint8_t*) malloc(image_data_size);
  read(images_fd, image_data, image_data_size);

  // INITIALIZE SET LOADER
  set = (set_loader_t*) malloc(sizeof(set_loader_t));
  set->idx = 0;
  set->total = num_images;
  set->width = width;
  set->height = height;
  set->data_size = width * height * num_images * sizeof(uint8_t);
  set->data = image_data;
  set->images = (image_t**) malloc(sizeof(image_t*)*num_images);
  for (int i = 0; i < num_images; i++) {
    image = (image_t*) malloc(sizeof(image_t));
    read(labels_fd, &(image->label), 1);
    assert(0 <= image->label && image->label <= 9);
    image->data = &(image_data[(width*height)*i]);
    set->images[i] = image;
  }
  set->access_order = (int*) malloc(sizeof(int) * num_images);
  for (int i = 0; i < num_images; i++) {
    set->access_order[i] = i;
  }
  close(labels_fd);
  close(images_fd);
  return set;
}

/*
  set_loader_free frees a set_loader_t
*/
void set_loader_free(set_loader_t *set) {
  for (int i = 0; i < set->total; i++) {
    free(set->images[i]);
  }
  free(set->access_order);
  free(set->images);
  free(set->data);
  free(set);
}

/*
  set_loader_free gets the next image_t from the set loader in
  sequential order from data file.
*/
image_t *get_next_image(set_loader_t *set) {
  image_t *img;
  if (set->idx >= set->total) return NULL; // reached the end
  img = set->images[set->access_order[set->idx++]];
  return img;
}

/*
  shuffle randomizes the access order of the images
*/
void shuffle(set_loader_t *set) {
  set->idx = 0;
  shuffle_index_array(set->access_order, set->total);
}

/*
  image_print displays an image_t
*/
void image_print(image_t *img, int dim) {
  printf("\nLabel: %d\n", (int)img->label);
  for (int i = 0; i < (dim*dim); i++) {
    if (i % dim == 0) {
      printf("%3d\n", (int)img->data[i]);
    } else {
      printf("%3d ", (int)img->data[i]);
    }
  }
  printf("%s\n", "");
}

// int main() {
//
//   set_loader_t *set;
//   if (verify_data()) {
//     printf("%s\n", "woohoo, verified");
//   } else {
//     printf("%s\n", "whoops, not verified");
//   }
//
//   set = init_set_loader(TRAIN_IMAGES, TRAIN_LABELS);
//   printf("%s\n", "No shuffle");
//   for (int i = 0; i < 5; i++) image_print(get_next_image(set), set->height);
//   shuffle(set);
//   printf("%s\n", "Post shuffle");
//   for (int i = 0; i < 5; i++) image_print(get_next_image(set), set->height);
//   set_loader_free(set);
//
//   return 0;
// }


// AUXILIARY FUNCTIONS

void swap(int *a, int b, int c) {
  int d = a[b];
  a[b] = a[c];
  a[c] = d;
}

void shuffle_index_array(int *arr, int size) {
    int tmp;
    // Fisher-Yates shuffle the indices.
    // http://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
    for (int i = size-1; i > 0; i--) {
        tmp = rand() % (i + 1);
        swap(arr, i, tmp);
    }
}
