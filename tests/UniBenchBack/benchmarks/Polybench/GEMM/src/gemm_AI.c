#include <omp.h>
#ifndef taskminerutils
#define taskminerutils
static int taskminer_depth_cutoff = 0;
#define DEPTH_CUTOFF omp_get_num_threads()
char cutoff_test = 0;
#endif
/**
 * gemm.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
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

/* Problem size */
#define NI 512
#define NJ 512
#define NK 512

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0)
 */
#define ALPHA 32412.0f
#define BETA 2123.0f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

#define GPU_DEVICE 1

void gemm(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C) {
  int i, j, k;

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < NI; i++) {
    {
    int tmc3 = 512 * (26);
    int tmc2 = 512 * (16 + tmc3);
    int tm_cost1 = (9 + tmc2);
    #pragma omp task depend(inout: A[0:262657],B[0:262657],C[0:262657]) if(tm_cost1 > 500)
    {
    for (j = 0; j < NJ; j++) {
      C[i * NJ + j] *= BETA;

      for (k = 0; k < NK; ++k) {
        C[i * NJ + j] += ALPHA * A[i * NK + k] * B[k * NJ + j];
      }
    }
  }
  }
  }
}

void gemm_OMP(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C) {
  int i, j, k;

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < NI; i++) {
    {
    int tmc3 = 512 * (26);
    int tmc2 = 512 * (16 + tmc3);
    int tm_cost1 = (9 + tmc2);
    #pragma omp task depend(inout: A[0:262657],B[0:262657],C[0:262657]) if(tm_cost1 > 500)
    {
    for (j = 0; j < NJ; j++) {
      C[i * NJ + j] *= BETA;

      for (k = 0; k < NK; ++k) {
        C[i * NJ + j] += ALPHA * A[i * NK + k] * B[k * NJ + j];
      }
    }
  }
  }
  }
}

void init(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *C_OMP) {
  int i, j;

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < NI; i++) {
    {
    int tmc6 = 512 * (16);
    int tm_cost5 = (9 + tmc6);
    #pragma omp task depend(inout: A[0:262657],B[0:262657],C[0:262657],C_OMP[0:262657]) if(tm_cost5 > 500)
    {
    for (j = 0; j < NK; j++) {
      A[i * NK + j] = ((DATA_TYPE)i * j) / NI;
    }
  }
  }
  }

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < NK; i++) {
    {
    int tmc4 = 512 * (17);
    int tm_cost3 = (9 + tmc4);
    #pragma omp task depend(inout: A[0:262657],B[0:262657],C[0:262657],C_OMP[0:262657]) if(tm_cost3 > 500)
    {
    for (j = 0; j < NJ; j++) {
      B[i * NJ + j] = ((DATA_TYPE)i * j + 1) / NJ;
    }
  }
  }
  }

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < NI; i++) {
    {
    int tmc2 = 512 * (27);
    int tm_cost1 = (9 + tmc2);
    #pragma omp task depend(inout: A[0:262657],B[0:262657],C[0:262657],C_OMP[0:262657]) if(tm_cost1 > 500)
    {
    for (j = 0; j < NJ; j++) {
      C[i * NJ + j] = ((DATA_TYPE)i * j + 2) / NJ;
      C_OMP[i * NJ + j] = ((DATA_TYPE)i * j + 2) / NJ;
    }
  }
  }
  }
}

void compareResults(DATA_TYPE *C, DATA_TYPE *C_outputFromGpu) {
  int i, j, fail;
  fail = 0;

  // Compare C1 and C2
  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < NI; i++) {
    {
    int tmc2 = 512 * (28);
    int tm_cost1 = (11 + tmc2);
    #pragma omp task depend(inout: C[0:262657],C_outputFromGpu[0:262657]) if(tm_cost1 > 500)
    {
    for (j = 0; j < NJ; j++) {
      if (percentDiff(C[i * NJ + j], C_outputFromGpu[i * NJ + j]) >
          PERCENT_DIFF_ERROR_THRESHOLD) {
        fail++;
      }
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
  DATA_TYPE *C;
  DATA_TYPE *C_outputFromGpu;

  A = (DATA_TYPE *)malloc(NI * NK * sizeof(DATA_TYPE));
  B = (DATA_TYPE *)malloc(NK * NJ * sizeof(DATA_TYPE));
  C = (DATA_TYPE *)malloc(NI * NJ * sizeof(DATA_TYPE));
  C_outputFromGpu = (DATA_TYPE *)malloc(NI * NJ * sizeof(DATA_TYPE));

  fprintf(stdout, "<< Matrix-multiply C=alpha.A.B+beta.C >>\n");

  init(A, B, C, C_outputFromGpu);

  t_start = rtclock();
  gemm_OMP(A, B, C_outputFromGpu);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  gemm(A, B, C);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(C, C_outputFromGpu);

  free(A);
  free(B);
  free(C);
  free(C_outputFromGpu);

  return 0;
}

