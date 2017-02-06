
#ifndef __NETWORK_H__
#define  __NETWORK_H__

#include "csapp.h"
#include <gsl/gsl_sf_bessel.h>


/* Recommended max cache and object sizes */
#define MAX_OBJECT_SIZE 102400
#define MAX_LAYERS 128

typedef struct network {
  int layers[MAX_LAYERS];

} network_t;

#endif
