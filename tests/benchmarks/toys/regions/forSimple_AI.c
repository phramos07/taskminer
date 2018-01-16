#include <omp.h>
#ifndef taskminerutils
#define taskminerutils
static int taskminer_depth_cutoff = 0;
#define DEPTH_CUTOFF omp_get_num_threads()
char cutoff_test = 0;
#endif
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#define SIZE 1000000
#define HOP 1000

int main() {
  int *results = (int *)malloc(sizeof(int) * SIZE);
  int *results_2 = (int *)malloc(sizeof(int) * SIZE);

  int *i;
  i = (int)malloc(sizeof(int));

  for (int j = HOP; j < SIZE; j += HOP) {
    int a = SIZE - 1;
    while (a > 1) {
      results[j] += results[j - 1] ^ 0x0000FFFF;
      a--;
    }

    *i += 2;

    for (int k = 0; k < SIZE; k++) {
      results_2[k] += results[k];
    }

    results_2[j] += results_2[j] + j + results[j];
  }

  int i_ = 0;
  while (i_ < SIZE - 1) {
    printf("%d\n", *i + results_2[*i]);
    *i++;
  }

  return 0;
}

