#include <omp.h>
#ifndef taskminerutils
#define taskminerutils
static int taskminer_depth_cutoff = 0;
#define DEPTH_CUTOFF omp_get_num_threads()
char cutoff_test = 0;
#endif
/**
 * atax.c: This file was adapted from PolyBench/GPU 1.0 test suite
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
// include <omp.h>

#include "../../common/polybenchUtilFuncts.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

/* Problem size. */
#define NX 8192
#define NY 8192

#define GPU_DEVICE 1

#ifndef M_PI
#define M_PI 3.14159
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE *x, DATA_TYPE *A) {
  int i, j;

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < NX; i++) {
    {
    int tmc2 = 8192 * (16);
    int tm_cost1 = (15 + tmc2);
    #pragma omp task depend(inout: A[0:67117057],x[0:8193]) if(tm_cost1 > 41)
    {
    x[i] = i * M_PI;
    for (j = 0; j < NY; j++) {
      A[i * NY + j] = ((DATA_TYPE)i * (j)) / NX;
    }
  }
  }
  }
}

void compareResults(DATA_TYPE *z, DATA_TYPE *z_outputFromGpu) {
  int i, fail;
  fail = 0;

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < NY; i++) {
    {
    int tm_cost1 = (24);
    #pragma omp task depend(inout: z[0:8193],z_outputFromGpu[0:8193]) if(tm_cost1 > 41)
    {
    if (percentDiff(z[i], z_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
      fail++;
    }
  }
  }
  }

  // print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
         "Percent: %d\n",
         PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void atax_cpu(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp) {
  int i, j;

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < NY; i++) {
    {
    int tm_cost4 = (10);
    #pragma omp task depend(inout: A[0:67117057],tmp[0:8193],x[0:8193],y[0:8193]) if(tm_cost4 > 41)
    {
    y[i] = 0;
  }
  }
  }

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < NX; i++) {
    {
    int tmc3 = 8192 * (23);
    int tmc2 = 8192 * (23);
    int tm_cost1 = (14 + tmc2 + tmc3);
    #pragma omp task depend(inout: A[0:67117057],tmp[0:8193],x[0:8193],y[0:8193]) if(tm_cost1 > 41)
    {
    tmp[i] = 0;

    for (j = 0; j < NY; j++) {
      tmp[i] = tmp[i] + A[i * NY + j] * x[j];
    }

    for (j = 0; j < NY; j++) {
      y[j] = y[j] + A[i * NY + j] * tmp[i];
    }
  }
  }
  }
}

void atax_OMP(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp) {
  int i, j;

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < NY; i++) {
    {
    int tm_cost5 = (10);
    #pragma omp task depend(inout: A[0:67117057],tmp[0:8193],x[0:8193],y[0:8193]) if(tm_cost5 > 41)
    {
    y[i] = 0;
  }
  }
  }

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < NX; i++) {
    {
    int tmc4 = 8192 * (23);
    int tm_cost3 = (12 + tmc4);
    #pragma omp task depend(inout: A[0:67117057],tmp[0:8193],x[0:8193],y[0:8193]) if(tm_cost3 > 41)
    {
    tmp[i] = 0;
    int j;
    for (j = 0; j < NY; j++) {
      tmp[i] = tmp[i] + A[i * NY + j] * x[j];
    }
  }
  }
  }

  // Note that the Loop has been reversed
  #pragma omp parallel
  #pragma omp single
  for (j = 0; j < NY; j++)
    {
    int tmc2 = 8192 * (23);
    int tm_cost1 = (9 + tmc2);
    #pragma omp task depend(inout: A[0:67117057],tmp[0:8193],x[0:8193],y[0:8193]) if(tm_cost1 > 41)
    {
    for (i = 0; i < NX; i++) {
      { y[j] = y[j] + A[i * NY + j] * tmp[i]; }
    }
    }
    }
}

int main(int argc, char **argv) {
  double t_start, t_end;

  DATA_TYPE *A;
  DATA_TYPE *x;
  DATA_TYPE *y;
  DATA_TYPE *y_outputFromGpu;
  DATA_TYPE *tmp;

  A = (DATA_TYPE *)malloc(NX * NY * sizeof(DATA_TYPE));
  x = (DATA_TYPE *)malloc(NY * sizeof(DATA_TYPE));
  y = (DATA_TYPE *)malloc(NY * sizeof(DATA_TYPE));
  y_outputFromGpu = (DATA_TYPE *)malloc(NY * sizeof(DATA_TYPE));
  tmp = (DATA_TYPE *)malloc(NX * sizeof(DATA_TYPE));

  fprintf(stdout, "<< Matrix Transpose and Vector Multiplication >>\n");

  init_array(x, A);

  t_start = rtclock();
  atax_OMP(A, x, y_outputFromGpu, tmp);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  atax_cpu(A, x, y, tmp);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(y, y_outputFromGpu);

  free(A);
  free(x);
  free(y);
  free(y_outputFromGpu);
  free(tmp);

  return 0;
}

