#ifndef __MNIST_LOADER_H__
#define  __MNIST_LOADER_H__

#include "../csapp.h"
#include <stdbool.h>

/**
MNIST Handrwitten Digits Dataset Loader

The training set contains 60000 examples, and the test set 10000 examples.

The first 5000 examples of the test set are taken from the original
NIST training set. The last 5000 are taken from the original NIST test set.
The first 5000 are cleaner and easier than the last 5000.

TRAINING SET LABEL FILE (train-labels-idx1-ubyte):

[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  60000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.

TRAINING SET IMAGE FILE (train-images-idx3-ubyte):

[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  60000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel
Pixels are organized row-wise. Pixel values are 0 to 255. 0 means
background (white), 255 means foreground (black).

TEST SET LABEL FILE (t10k-labels-idx1-ubyte):

[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  10000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.

TEST SET IMAGE FILE (t10k-images-idx3-ubyte):

[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  10000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel
Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background
(white), 255 means foreground (black).
**/


// data files and respective magic numbers

#define TRAIN_IMAGES "data/train-images.idx3-ubyte"
#define TRAIN_IMAGES_MN 0x00000803

#define TRAIN_LABELS "data/train-labels.idx1-ubyte"
#define TRAIN_LABELS_MN 0x00000801

#define TEST_IMAGES "data/t10k-images.idx3-ubyte"
#define TEST_IMAGES_MN 0x00000803

#define TEST_LABELS "data/t10k-labels.idx1-ubyte"
#define TEST_LABELS_MN 0x00000801

typedef struct image {
  uint8_t *data;
  uint8_t label;
} image_t;

typedef struct set_loader {
  size_t idx;         // current image
  size_t total;       // total number of images
  uint8_t *data;      // image data
  image_t **images;   // images structures
  int *access_order;  // randomized indeces
  size_t data_size;   // data size
  size_t height;
  size_t width;
} set_loader_t;

bool verify_data();
set_loader_t *init_set_loader(const char *data_file, const char *label_file);
void set_loader_free(set_loader_t *set);
image_t *get_next_image(set_loader_t *set);
void shuffle(set_loader_t *set);
void image_print(image_t *img, int dim);

// auxiliary functions
void swap(int *a, int b, int c);
void shuffle_index_array(int *arr, int size);

#endif
