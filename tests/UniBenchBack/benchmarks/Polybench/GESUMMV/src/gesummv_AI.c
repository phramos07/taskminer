#include <omp.h>
#ifndef taskminerutils
#define taskminerutils
static int taskminer_depth_cutoff = 0;
#define DEPTH_CUTOFF omp_get_num_threads()
char cutoff_test = 0;
#endif
/**
 * gesummv.c: This file was adapted from PolyBench/GPU 1.0 test
 * suite to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *           Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
// include <omp.h>

#include "../../common/polybenchUtilFuncts.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 1

/* Problem size */
#define N 8192

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0)
 */
#define ALPHA 43532.0f
#define BETA 12313.0f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void gesummv(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x, DATA_TYPE *y,
             DATA_TYPE *tmp) {
  int i, j;

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < N; i++) {
    {
    int tmc2 = 8192 * (39);
    int tm_cost1 = (27 + tmc2);
    #pragma omp task depend(inout: A[0:67117057],B[0:67117057],tmp[0:8193],x[0:8193],y[0:8193]) if(tm_cost1 > 500)
    {
    tmp[i] = 0;
    y[i] = 0;
    for (j = 0; j < N; j++) {
      tmp[i] = A[i * N + j] * x[j] + tmp[i];
      y[i] = B[i * N + j] * x[j] + y[i];
    }

    y[i] = ALPHA * tmp[i] + BETA * y[i];
  }
  }
  }
}

void gesummv_OMP(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x, DATA_TYPE *y,
                 DATA_TYPE *tmp) {
  int i, j;

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < N; i++) {
    {
    int tmc2 = 8192 * (39);
    int tm_cost1 = (27 + tmc2);
    #pragma omp task depend(inout: A[0:67117057],B[0:67117057],tmp[0:8193],x[0:8193],y[0:8193]) if(tm_cost1 > 500)
    {
    tmp[i] = 0;
    y[i] = 0;
    for (j = 0; j < N; j++) {
      tmp[i] = A[i * N + j] * x[j] + tmp[i];
      y[i] = B[i * N + j] * x[j] + y[i];
    }

    y[i] = ALPHA * tmp[i] + BETA * y[i];
  }
  }
  }
}

void init(DATA_TYPE *A, DATA_TYPE *x) {
  int i, j;

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < N; i++) {
    {
    int tmc2 = 8192 * (16);
    int tm_cost1 = (14 + tmc2);
    #pragma omp task depend(inout: A[0:67117057],x[0:8193]) if(tm_cost1 > 500)
    {
    x[i] = ((DATA_TYPE)i) / N;

    for (j = 0; j < N; j++) {
      A[i * N + j] = ((DATA_TYPE)i * j) / N;
    }
  }
  }
  }
}

void compareResults(DATA_TYPE *y, DATA_TYPE *y_outputFromGpu) {
  int i, fail;
  fail = 0;

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < (N); i++) {
    {
    int tm_cost1 = (24);
    #pragma omp task depend(inout: y[0:8193],y_outputFromGpu[0:8193]) if(tm_cost1 > 500)
    {
    if (percentDiff(y[i], y_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
      fail++;
    }
  }
  }
  }

  // Print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
         "Percent: %d\n",
         PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

int main(int argc, char *argv[]) {
  double t_start, t_end;

  DATA_TYPE *A;
  DATA_TYPE *B;
  DATA_TYPE *x;
  DATA_TYPE *y;
  DATA_TYPE *y_outputFromGpu;
  DATA_TYPE *tmp;

  A = (DATA_TYPE *)malloc(N * N * sizeof(DATA_TYPE));
  B = (DATA_TYPE *)malloc(N * N * sizeof(DATA_TYPE));
  x = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  y = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  y_outputFromGpu = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  tmp = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));

  fprintf(stdout, "<< Scalar, Vector and Matrix Multiplication >>\n");

  init(A, x);

  t_start = rtclock();
  gesummv_OMP(A, B, x, y_outputFromGpu, tmp);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  gesummv(A, B, x, y, tmp);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(y, y_outputFromGpu);

  free(A);
  free(B);
  free(x);
  free(y);
  free(y_outputFromGpu);
  free(tmp);

  return 0;
}

