#include <omp.h>
#ifndef taskminerutils
#define taskminerutils
static int taskminer_depth_cutoff = 0;
#define DEPTH_CUTOFF omp_get_num_threads()
char cutoff_test = 0;
#endif
#include <omp.h>
#include <stdlib.h>

int sum_range(int *V, int N, int L, int U, int *A) {
  int i = 0, sum = 0;
  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < N; i++) {
    {
    long long int TM14[2];
    TM14[0] = L > 0;
    TM14[1] = (TM14[0] ? L : 0);
    int tmc2 = TM14[1] * (15);
    int tm_cost1 = (18 + tmc2);
    long long int TM13[14];
    TM13[0] = L > 0;
    TM13[1] = (TM13[0] ? L : 0);
    TM13[2] = 4 * TM13[1];
    TM13[3] = TM13[2] + 4;
    TM13[4] = TM13[3] / 4;
    TM13[5] = (TM13[4] > 0);
    TM13[6] = (TM13[5] ? TM13[4] : 0);
    TM13[7] = N > 0;
    TM13[8] = (TM13[7] ? N : 0);
    TM13[9] = 4 * TM13[8];
    TM13[10] = TM13[9] + 4;
    TM13[11] = TM13[10] / 4;
    TM13[12] = (TM13[11] > 0);
    TM13[13] = (TM13[12] ? TM13[11] : 0);
    #pragma omp task depend(inout: A[0:TM13[13]],V[0:TM13[6]]) if(tm_cost1 > 6000)
    {
    int j = 0; // V[i];
    A[i] = 0;
    for (; j < L; j++) {
      A[i] += V[j];
      // j++;
    }
    sum += A[i];
  }
  }
  }

  return sum;
}

int main(int argc, char const *argv[]) { return 0; }

