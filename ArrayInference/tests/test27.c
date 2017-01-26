#include <stdlib.h>
#include <stdio.h>

int main(int argc, char **argv) {
  int N = atoi(argv[1]);
  int *array;
  array = (int *)malloc(20 * sizeof(int));

  for (int i = 0; i < N; i++) {
    array[i] = i;
  }

  return 0;
}
