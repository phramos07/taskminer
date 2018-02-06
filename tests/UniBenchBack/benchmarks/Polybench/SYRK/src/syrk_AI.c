#include <omp.h>
#ifndef taskminerutils
#define taskminerutils
static int taskminer_depth_cutoff = 0;
#define DEPTH_CUTOFF omp_get_num_threads()
char cutoff_test = 0;
#endif
/**
 * syrk.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *	     Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include "../../common/polybenchUtilFuncts.h"

// define the error threshold for the results "not matching"
#define ERROR_THRESHOLD 0.05
#define GPU_DEVICE 1

/* Problem size */
#define N 1024
#define M 1024

/* Declared constant values for alpha and beta */
/* (same as values in PolyBench 2.0) */
#define alpha 12435
#define beta 4546

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_arrays(DATA_TYPE *A, DATA_TYPE *C, DATA_TYPE *D) {
  int i, j;

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < N; i++) {
    {
    int tmc3 = 1024 * (27);
    int tmc2 = 1024 * (16);
    int tm_cost1 = (11 + tmc2 + tmc3);
    #pragma omp task depend(inout: A[0:1049601],C[0:1049601],D[0:1049601]) if(tm_cost1 > 41)
    {
    for (j = 0; j < M; j++) {
      A[i * M + j] = ((DATA_TYPE)i * j) / N;
    }
    for (j = 0; j < M; j++) {
      C[i * M + j] = ((DATA_TYPE)i * j + 2) / N;
      D[i * M + j] = ((DATA_TYPE)i * j + 2) / N;
    }
  }
  }
  }
}

void compareResults(DATA_TYPE *C, DATA_TYPE *D) {
  int i, j, fail;
  fail = 0;

  // Compare C with D
  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < N; i++) {
    {
    int tmc2 = 1024 * (28);
    int tm_cost1 = (11 + tmc2);
    #pragma omp task depend(inout: C[0:1049601],D[0:1049601]) if(tm_cost1 > 41)
    {
    for (j = 0; j < M; j++) {
      if (percentDiff(C[i * M + j], D[i * M + j]) > ERROR_THRESHOLD) {
        fail++;
      }
    }
  }
  }
  }

  // print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
         "Percent: %d\n",
         ERROR_THRESHOLD, fail);
}

void syrk(DATA_TYPE *A, DATA_TYPE *C) {
  int i, j, k;

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < N; i++) {
    {
    int tmc5 = 1024 * (14);
    int tm_cost4 = (9 + tmc5);
    #pragma omp task depend(inout: A[0:1049601],C[0:1049601]) if(tm_cost4 > 41)
    {
    for (j = 0; j < M; j++) {
      C[i * M + j] *= beta;
    }
  }
  }
  }

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < N; i++) {
    {
    int tmc3 = 1024 * (26);
    int tmc2 = 1024 * (9 + tmc3);
    int tm_cost1 = (9 + tmc2);
    #pragma omp task depend(inout: A[0:1049601],C[0:1049601]) if(tm_cost1 > 41)
    {
    for (j = 0; j < M; j++) {
      for (k = 0; k < M; k++) {
        C[i * N + j] += alpha * A[i * M + k] * A[j * M + k];
      }
    }
  }
  }
  }
}

void syrkGPU(DATA_TYPE *A, DATA_TYPE *D) {
  int i, j;
  double t_start, t_end;

  t_start = rtclock();

  {
    #pragma omp parallel
    #pragma omp single
    for (i = 0; i < N; i++) {
      {
      int tmc5 = 1024 * (14);
      int tm_cost4 = (14 + tmc5);
      #pragma omp task depend(inout: D[0:1049601]) if(tm_cost4 > 41)
      {
      {
      int tmc5 = 1024 * (14);
      int tm_cost4 = (19 + tmc5);
      #pragma omp task depend(inout: D[0:1049601]) if(tm_cost4 > 41)
      {
      for (j = 0; j < M; j++) {
        D[i * M + j] *= beta;
      }
    }
    }
    }

    #pragma omp parallel
    #pragma omp single
    for (i = 0; i < N; i++) {
      {
      int tmc3 = 1024 * (26);
      int tmc2 = 1024 * (9 + tmc3);
      int tm_cost1 = (19 + tmc2);
      #pragma omp task depend(inout: A[0:1049601],D[0:1049601]) if(tm_cost1 > 41)
      {
      {
      int tmc3 = 1024 * (26);
      int tmc2 = 1024 * (9 + tmc3);
      int tm_cost1 = (29 + tmc2);
      #pragma omp task depend(inout: A[0:1049601],D[0:1049601]) if(tm_cost1 > 41)
      {
      for (j = 0; j < M; j++) {
        int k;
        for (k = 0; k < M; k++) {
          D[i * M + j] += alpha * A[i * M + k] * A[j * M + k];
        }
      }
    }
    }
    }
  }

  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
}

int main() {
  double t_start, t_end;

  DATA_TYPE *A;
  DATA_TYPE *C;
  DATA_TYPE *D;

  A = (DATA_TYPE *)malloc(N * M * sizeof(DATA_TYPE));
  C = (DATA_TYPE *)malloc(N * M * sizeof(DATA_TYPE));
  D = (DATA_TYPE *)malloc(N * M * sizeof(DATA_TYPE));

  fprintf(stdout, "<< Symmetric rank-k operations >>\n");

  init_arrays(A, C, D);
  syrkGPU(A, D);

  t_start = rtclock();
  syrk(A, C);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(C, D);

  free(A);
  free(C);
  free(D);
  return 0;
}

