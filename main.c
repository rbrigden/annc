#include "network.h"

int main() {
  gsl_matrix *m1;

  printf("%s\n", "Beginning script");
  m1 = rand_gaussian_matrix(10,10);
  print_matrix (stdout, m1);

  return 0;
}
