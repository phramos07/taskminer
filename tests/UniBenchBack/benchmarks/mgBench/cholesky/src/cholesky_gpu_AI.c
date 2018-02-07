#include <omp.h>
#ifndef taskminerutils
#define taskminerutils
static int taskminer_depth_cutoff = 0;
#define DEPTH_CUTOFF omp_get_num_threads()
char cutoff_test = 0;
#endif
/*
   This program performs cholesky decomposition on the GPU with
   dynamically allocated matrices.

    Author: Gleison Souza Diniz Mendon?a
    Date: 04-01-2015
    version 2.0

    Run:
    ipmacc cholesky_gpu.c -o cholesky
    ./cholesky matrix-size
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include "../../common/mgbenchUtilFunctions.h"

#define SIZE 1000
#define GPU_DEVICE 0
#define ERROR_THRESHOLD 0.05

// Initialize matrices.
void init_arrays(float *A, float *B_GPU, float *B_CPU) {
  int i, j, q;
  q = SIZE * SIZE;

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < SIZE; ++i) {
    {
    int tmc2 = 1000 * (27);
    int tm_cost1 = (9 + tmc2);
    #pragma omp task depend(inout: A[0:1001001],B_CPU[0:1001001],B_GPU[0:1001001]) if(tm_cost1 > 41)
    {
    for (j = 0; j < SIZE; ++j) {
      A[i * SIZE + j] = (float)(q - (10 * i) - (5 * j));
      B_GPU[i * SIZE + j] = 0.0f;
      B_CPU[i * SIZE + j] = 0.0f;
    }
  }
  }
  }
}

/// Cholesky algorithm GPU
/// s = size of matrix
void cholesky_GPU(float *A, float *B) {
  int i, j, k, l;

  float t;

  {
    for (i = 0; i < SIZE; i++) {
      for (j = 0; j <= i; j++) {
        t = 0.0f;
        for (k = 0; k < j; k++) {
          if (B[i * SIZE + k] != 0.0f && B[j * SIZE + k] != 0.0f) {
            t += B[i * SIZE + k] * B[j * SIZE + k];
          } else {
            k--;
          }
        }
        if (i == j) {
          B[i * SIZE + j] = sqrt((A[i * SIZE + i] - t));
        } else {
          if (B[j * SIZE + j] != 0.0f) {
            B[i * SIZE + j] = (1.0 / B[j * SIZE + j] * (A[i * SIZE + j] - t));
          } else {
            j--;
          }
        }
      }
    }
  }
}

void cholesky_CPU(float *A, float *B) {
  int i, j, k;

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < SIZE; i++) {
    {
    int tmc3 = 1001 * (21);
    int tmc2 = 1001 * (50 + tmc3);
    int tm_cost1 = (9 + tmc2);
    #pragma omp task depend(inout: A[0:1001002],B[0:1002002]) if(tm_cost1 > 41)
    {
    for (j = 0; j <= i; j++) {
      float t;
      t = 0.0f;
      for (k = 0; k < j; k++) {
        t += B[i * SIZE + k] * B[j * SIZE + k];
      }
      if (i == j) {
        B[i * SIZE + j] = sqrt((A[i * SIZE + i] - t));
      } else {
        B[i * SIZE + j] = (1.0 / B[j * SIZE + j] * (A[i * SIZE + j] - t));
      }
    }
  }
  }
  }
}

void compareResults(float *E, float *E_GPU) {
  int i, j, fail;
  fail = 0;

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < SIZE; i++) {
    {
    int tmc2 = 1000 * (28);
    int tm_cost1 = (11 + tmc2);
    #pragma omp task depend(inout: E[0:1001001],E_GPU[0:1001001]) if(tm_cost1 > 41)
    {
    for (j = 0; j < SIZE; j++) {
      if (percentDiff(E[i * SIZE + j], E_GPU[i * SIZE + j]) > ERROR_THRESHOLD) {
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

int main(int argc, char *argv[]) {
  double t_start, t_end;
  float *A, *B_CPU, *B_GPU;

  A = (float *)malloc(SIZE * SIZE * sizeof(float));
  B_CPU = (float *)malloc(SIZE * SIZE * sizeof(float));
  B_GPU = (float *)malloc(SIZE * SIZE * sizeof(float));

  fprintf(stdout, "<< Cholesky >>\n");

  init_arrays(A, B_CPU, B_GPU);

  t_start = rtclock();
  cholesky_GPU(A, B_GPU);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  cholesky_CPU(A, B_CPU);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(B_CPU, B_GPU);

  free(A);
  free(B_CPU);
  free(B_GPU);

  return 0;
}

