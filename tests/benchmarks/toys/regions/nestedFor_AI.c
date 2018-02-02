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

  #pragma omp parallel
  #pragma omp single
  for (int a = HOP; a < SIZE; a += HOP) {
    int tmc2 = 10 * (28);
    int tmc3 = 1000000 * (15);
    int tm_cost1 = (44 + tmc2 + tmc3);
    #pragma omp task depend(inout: results[999:999001],results_2[1000:999000]) if(tm_cost1 > 1000)
    {
    results[a] = 0;
    for (int j = 1; j < SIZE; j++) {
      results[a] += j;
      results[a] += results[a - 1] || 0x0000FFFF + j;
      // results[a] += results[a-HOP] && 0x0000FFFF + j;
    }

    for (int j = 0; j < SIZE; j++)
      results_2[a] += results[a];
  }
  }

  return 0;
}

