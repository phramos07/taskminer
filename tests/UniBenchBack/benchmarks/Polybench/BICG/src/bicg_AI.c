#include <omp.h>
#ifndef taskminerutils
#define taskminerutils
static int taskminer_depth_cutoff = 0;
#define DEPTH_CUTOFF omp_get_num_threads()
char cutoff_test = 0;
#endif
/**
 * bicg.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *           Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
// include <omp.h>

#include "../../common/polybenchUtilFuncts.h"

// Error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.7

/* Problem size. */
#define NX 8192
#define NY 8192

#define GPU_DEVICE 1

#ifndef M_PI
#define M_PI 3.14159
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *r) {
  int i, j;

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < NX; i++) {
    {
    int tmc3 = 8192 * (16);
    int tm_cost2 = (15 + tmc3);
    #pragma omp task depend(inout: A[0:67117057],p[0:8193],r[0:8193]) if(tm_cost2 > 500)
    {
    r[i] = i * M_PI;
    for (j = 0; j < NY; j++) {
      A[i * NY + j] = ((DATA_TYPE)i * j) / NX;
    }
  }
  }
  }

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < NY; i++) {
    {
    int tm_cost1 = (13);
    #pragma omp task depend(inout: A[0:67117057],p[0:8193],r[0:8193]) if(tm_cost1 > 500)
    {
    p[i] = i * M_PI;
  }
  }
  }
}

void compareResults(DATA_TYPE *s, DATA_TYPE *s_outputFromGpu, DATA_TYPE *q,
                    DATA_TYPE *q_outputFromGpu) {
  int i, fail;
  fail = 0;

  // Compare s with s_cuda
  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < NX; i++) {
    {
    int tm_cost2 = (24);
    #pragma omp task depend(inout: q[0:8193],q_outputFromGpu[0:8193]) if(tm_cost2 > 500)
    {
    if (percentDiff(q[i], q_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
      fail++;
    }
  }
  }
  }

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < NY; i++) {
    {
    int tm_cost1 = (24);
    #pragma omp task depend(inout: s[0:8193],s_outputFromGpu[0:8193]) if(tm_cost1 > 500)
    {
    if (percentDiff(s[i], s_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
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

void bicg_cpu(DATA_TYPE *A, DATA_TYPE *r, DATA_TYPE *s, DATA_TYPE *p,
              DATA_TYPE *q) {
  int i, j;

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < NY; i++) {
    {
    int tm_cost3 = (10);
    #pragma omp task depend(inout: A[0:67117057],p[0:8193],q[0:8193],r[0:8193],s[0:8193]) if(tm_cost3 > 500)
    {
    s[i] = 0.0;
  }
  }
  }

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < NX; i++) {
    {
    int tmc2 = 8192 * (39);
    int tm_cost1 = (12 + tmc2);
    #pragma omp task depend(inout: A[0:67117057],p[0:8193],q[0:8193],r[0:8193],s[0:8193]) if(tm_cost1 > 500)
    {
    q[i] = 0.0;
    for (j = 0; j < NY; j++) {
      s[j] = s[j] + r[i] * A[i * NY + j];
      q[i] = q[i] + A[i * NY + j] * p[j];
    }
  }
  }
  }
}

void bicg_OMP(DATA_TYPE *A, DATA_TYPE *r, DATA_TYPE *s, DATA_TYPE *p,
              DATA_TYPE *q) {
  int i, j;

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < NY; i++) {
    {
    int tm_cost5 = (10);
    #pragma omp task depend(inout: A[0:67117057],p[0:8193],q[0:8193],r[0:8193],s[0:8193]) if(tm_cost5 > 500)
    {
    s[i] = 0.0;
  }
  }
  }

  #pragma omp parallel
  #pragma omp single
  for (j = 0; j < NY; j++) {
    {
    int tmc4 = 8192 * (23);
    int tm_cost3 = (9 + tmc4);
    #pragma omp task depend(inout: A[0:67117057],p[0:8193],q[0:8193],r[0:8193],s[0:8193]) if(tm_cost3 > 500)
    {
    for (i = 0; i < NX; i++) {
      s[j] = s[j] + r[i] * A[i * NY + j];
    }
  }
  }
  }

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < NX; i++) {
    {
    int tmc2 = 8192 * (23);
    int tm_cost1 = (12 + tmc2);
    #pragma omp task depend(inout: A[0:67117057],p[0:8193],q[0:8193],r[0:8193],s[0:8193]) if(tm_cost1 > 500)
    {
    q[i] = 0.0;
    for (j = 0; j < NY; j++) {
      q[i] = q[i] + A[i * NY + j] * p[j];
    }
  }
  }
  }
}

int main(int argc, char **argv) {
  double t_start, t_end;

  DATA_TYPE *A;
  DATA_TYPE *r;
  DATA_TYPE *s;
  DATA_TYPE *p;
  DATA_TYPE *q;
  DATA_TYPE *s_GPU;
  DATA_TYPE *q_GPU;

  A = (DATA_TYPE *)malloc(NX * NY * sizeof(DATA_TYPE));
  r = (DATA_TYPE *)malloc(NX * sizeof(DATA_TYPE));
  s = (DATA_TYPE *)malloc(NY * sizeof(DATA_TYPE));
  p = (DATA_TYPE *)malloc(NY * sizeof(DATA_TYPE));
  q = (DATA_TYPE *)malloc(NX * sizeof(DATA_TYPE));
  s_GPU = (DATA_TYPE *)malloc(NY * sizeof(DATA_TYPE));
  q_GPU = (DATA_TYPE *)malloc(NX * sizeof(DATA_TYPE));

  fprintf(stdout, "<< BiCG Sub Kernel of BiCGStab Linear Solver >>\n");

  init_array(A, p, r);

  t_start = rtclock();
  bicg_OMP(A, r, s_GPU, p, q_GPU);
  t_end = rtclock();

  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  bicg_cpu(A, r, s, p, q);
  t_end = rtclock();

  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(s, s_GPU, q, q_GPU);

  free(A);
  free(r);
  free(s);
  free(p);
  free(q);
  free(s_GPU);
  free(q_GPU);

  return 0;
}

