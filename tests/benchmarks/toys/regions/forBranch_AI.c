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
#define HOP 100

int main() {
  int *results = (int *)malloc(sizeof(int) * SIZE);
  int *results_2 = (int *)malloc(sizeof(int) * SIZE);
  int i, a;

  #pragma omp parallel
  #pragma omp single
  for (int j = HOP; j < SIZE; j += HOP) {
    #pragma omp task depend(inout: results[100:999900],results_2[100:999900])
    {
    if (j % 2) {
      for (int a = 0; a < SIZE; a++) {
        results[j] += results_2[j];
      }
    } else {
      for (int a = 0; a < SIZE; a++) {
        results_2[j] += results[j];
      }
    }
  }
  }

  return 0;
}

