#include <omp.h>
#ifndef taskminerutils
#define taskminerutils
static int taskminer_depth_cutoff = 0;
#define DEPTH_CUTOFF omp_get_num_threads()
char cutoff_test = 0;
#endif
/**
 * 2DConvolution.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *	     Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
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
#define ERROR_THRESHOLD 0.05

#define GPU_DEVICE 1

/* Problem size */
#define NI 8192
#define NJ 8192

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void conv2D(DATA_TYPE *A, DATA_TYPE *B) {
  int i, j;
  DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

  c11 = +0.2;
  c21 = +0.5;
  c31 = -0.8;
  c12 = -0.3;
  c22 = +0.6;
  c32 = -0.9;
  c13 = +0.4;
  c23 = +0.7;
  c33 = +0.10;

  #pragma omp parallel
  #pragma omp single
  for (i = 1; i < NI - 1; ++i) {
    {
    int tmc2 = 100 * (92);
    int tm_cost1 = (9 + tmc2);
    #pragma omp task depend(inout: A[0:67117057],B[8193:67100671]) if(tm_cost1 > 6000)
    {
    for (j = 1; j < NJ - 1; ++j) {
      B[i * NJ + j] =
          c11 * A[(i - 1) * NJ + (j - 1)] + c12 * A[(i + 0) * NJ + (j - 1)] +
          c13 * A[(i + 1) * NJ + (j - 1)] + c21 * A[(i - 1) * NJ + (j + 0)] +
          c22 * A[(i + 0) * NJ + (j + 0)] + c23 * A[(i + 1) * NJ + (j + 0)] +
          c31 * A[(i - 1) * NJ + (j + 1)] + c32 * A[(i + 0) * NJ + (j + 1)] +
          c33 * A[(i + 1) * NJ + (j + 1)];
    }
  }
  }
  }
}

void conv2D_OMP(DATA_TYPE *A, DATA_TYPE *B) {
  int i, j;
  DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

  c11 = +0.2;
  c21 = +0.5;
  c31 = -0.8;
  c12 = -0.3;
  c22 = +0.6;
  c32 = -0.9;
  c13 = +0.4;
  c23 = +0.7;
  c33 = +0.10;

  #pragma omp parallel
  #pragma omp single
  for (i = 1; i < NI - 1; ++i) {
    {
    int tmc2 = 100 * (92);
    int tm_cost1 = (9 + tmc2);
    #pragma omp task depend(inout: A[0:67117057],B[8193:67100671]) if(tm_cost1 > 6000)
    {
    for (j = 1; j < NJ - 1; ++j) {
      B[i * NJ + j] =
          c11 * A[(i - 1) * NJ + (j - 1)] + c12 * A[(i + 0) * NJ + (j - 1)] +
          c13 * A[(i + 1) * NJ + (j - 1)] + c21 * A[(i - 1) * NJ + (j + 0)] +
          c22 * A[(i + 0) * NJ + (j + 0)] + c23 * A[(i + 1) * NJ + (j + 0)] +
          c31 * A[(i - 1) * NJ + (j + 1)] + c32 * A[(i + 0) * NJ + (j + 1)] +
          c33 * A[(i + 1) * NJ + (j + 1)];
    }
  }
  }
  }
}

void init(DATA_TYPE *A) {
  int i, j;

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < NI; ++i) {
    {
    int tmc2 = 8192 * (15);
    int tm_cost1 = (9 + tmc2);
    #pragma omp task depend(inout: A[0:67117057]) if(tm_cost1 > 6000)
    {
    for (j = 0; j < NJ; ++j) {
      A[i * NJ + j] = (float)rand() / RAND_MAX;
    }
  }
  }
  }
}

void compareResults(DATA_TYPE *B, DATA_TYPE *B_GPU) {
  int i, j, fail;
  fail = 0;

  // Compare B and B_GPU
  #pragma omp parallel
  #pragma omp single
  for (i = 1; i < (NI - 1); i++) {
    {
    int tmc2 = 100 * (28);
    int tm_cost1 = (11 + tmc2);
    #pragma omp task depend(inout: B[8193:67100671],B_GPU[8193:67100671]) if(tm_cost1 > 6000)
    {
    for (j = 1; j < (NJ - 1); j++) {
      if (percentDiff(B[i * NJ + j], B_GPU[i * NJ + j]) > ERROR_THRESHOLD) {
        fail++;
      }
    }
  }
  }
  }

  // Print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
         "Percent: %d\n",
         ERROR_THRESHOLD, fail);
}

int main(int argc, char *argv[]) {
  double t_start, t_end, t_start_OMP, t_end_OMP;

  DATA_TYPE *A;
  DATA_TYPE *B;
  DATA_TYPE *B_OMP;

  A = (DATA_TYPE *)malloc(NI * NJ * sizeof(DATA_TYPE));
  B = (DATA_TYPE *)malloc(NI * NJ * sizeof(DATA_TYPE));
  B_OMP = (DATA_TYPE *)malloc(NI * NJ * sizeof(DATA_TYPE));

  fprintf(stdout, ">> Two dimensional (2D) convolution <<\n");

  // initialize the arrays
  init(A);

  t_start_OMP = rtclock();
  conv2D_OMP(A, B_OMP);
  t_end_OMP = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end_OMP - t_start_OMP); //);

  t_start = rtclock();
  conv2D(A, B);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start); //);

  compareResults(B, B_OMP);

  free(A);
  free(B);
  free(B_OMP);

  return 0;
}

