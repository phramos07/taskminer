#include <omp.h>
#ifndef taskminerutils
#define taskminerutils
static int taskminer_depth_cutoff = 0;
#define DEPTH_CUTOFF omp_get_num_threads()
char cutoff_test = 0;
#endif
/*
    Analisar porque SIZE>=419 gera erro de falha de segmentação


    This program calculates the distance between the k neighbors in a Cartesian
   map.
    It generates a matrix with the distance between the neighbors.
    This program create a csv file with the time execution results for each
   function(CPU,GPU) in this format: size of matrix, cpu time, gpu time.

    Author: Gleison Souza Diniz Mendonça
    Date: 04-01-2015
    version 1.0

    Run:
    folder_ipmacc/ipmacc folder_archive/k-nearest.c
    ./a.out
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include "../../common/mgbenchUtilFunctions.h"

#define SIZE 400
#define GPU_DEVICE 0
#define ERROR_THRESHOLD 0.05

/// initialize the cartesian map
void init(int *matrix, int *matrix_dist_cpu, int *matrix_dist_gpu) {
  int i, j, r, m;
  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < SIZE; i++) {
    {
    int tmc4 = 400 * (22);
    int tm_cost3 = (9 + tmc4);
    #pragma omp task depend(inout: matrix[0:160401],matrix_dist_cpu[0:160401],matrix_dist_gpu[0:160401]) if(tm_cost3 > 41)
    {
    for (j = 0; j < SIZE; j++) {
      matrix[i * SIZE + j] = 99999999;
      matrix_dist_cpu[i * SIZE + j] = 99999999;
      matrix_dist_gpu[i * SIZE + j] = 99999999;
    }
  }
  }
  }

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < SIZE; i++) {
    {
    int tmc2 = TM15[3] * (24);
    int tm_cost1 = (12 + tmc2);
    #pragma omp task depend(inout: matrix[0:160401],matrix_dist_cpu[0:160401],matrix_dist_gpu[0:160401]) if(tm_cost1 > 41)
    {
    {
    int tmc2 = TM15[3] * (28);
    int tm_cost1 = (12 + tmc2);
    #pragma omp task depend(inout: matrix[0:160401],matrix_dist_cpu[0:160401],matrix_dist_gpu[0:160401]) if(tm_cost1 > 41)
    {
    r = (i * 97) % SIZE;
    for (j = 0; j < r; j++) {
      matrix[i * SIZE + j] = (((j * 1021) * 71 % (SIZE * SIZE)) + 1);
      ;
      if (i == j) {
        matrix[i * SIZE + j] = 0;
      }
    }
  }
  }
  }
}

/// Knearest algorithm GPU
/// s = size of cartesian map
void Knearest_GPU(int *matrix, int *matrix_dist) {
  int i, j, k;
  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < SIZE; i++) {
    {
    int tmc5 = 400 * (25);
    int tm_cost4 = (14 + tmc5);
    #pragma omp task depend(inout: matrix[0:160401],matrix_dist[0:160401]) if(tm_cost4 > 41)
    {
    for (j = 0; j < SIZE; j++) {
      if (matrix[i * SIZE + j] != 99999999) {
        matrix_dist[i * SIZE + j] = matrix[i * SIZE + j];
      }
    }
    matrix_dist[i * SIZE + i] = 0;
  }
  }
  }

  /// opportunity of parallelism here
  {
    #pragma omp parallel
    #pragma omp single
    for (i = 0; i < SIZE; i++) {
      {
      int tmc3 = 400 * (56);
      int tmc2 = 400 * (9 + tmc3);
      int tm_cost1 = (9 + tmc2);
      #pragma omp task depend(inout: matrix[0:160401],matrix_dist[0:160401]) if(tm_cost1 > 41)
      {
      for (k = 0; k < SIZE; k++) {
        for (j = 0; j < SIZE; j++) {
          if (matrix_dist[k * SIZE + i] != 99999999 &&
              matrix_dist[i * SIZE + j] != 99999999 &&
              matrix_dist[k * SIZE + j] >
                  matrix_dist[k * SIZE + i] + matrix_dist[i * SIZE + j]) {
            matrix_dist[k * SIZE + j] =
                matrix_dist[k * SIZE + i] + matrix_dist[i * SIZE + j];
          }
        }
      }
    }
    }
    }
  }
}

void Knearest_CPU(int *matrix, int *matrix_dist) {
  int i, j, k;
  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < SIZE; i++) {
    {
    int tmc5 = 400 * (25);
    int tm_cost4 = (14 + tmc5);
    #pragma omp task depend(inout: matrix[0:160401],matrix_dist[0:160401]) if(tm_cost4 > 41)
    {
    for (j = 0; j < SIZE; j++) {
      if (matrix[i * SIZE + j] != 99999999) {
        matrix_dist[i * SIZE + j] = matrix[i * SIZE + j];
      }
    }
    matrix_dist[i * SIZE + i] = 0;
  }
  }
  }

  /// opportunity of parallelism here
  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < SIZE; i++) {
    {
    int tmc3 = 400 * (56);
    int tmc2 = 400 * (9 + tmc3);
    int tm_cost1 = (9 + tmc2);
    #pragma omp task depend(inout: matrix[0:160401],matrix_dist[0:160401]) if(tm_cost1 > 41)
    {
    for (k = 0; k < SIZE; k++) {
      for (j = 0; j < SIZE; j++) {
        if (matrix_dist[k * SIZE + i] != 99999999 &&
            matrix_dist[i * SIZE + j] != 99999999 &&
            matrix_dist[k * SIZE + j] >
                matrix_dist[k * SIZE + i] + matrix_dist[i * SIZE + j]) {
          matrix_dist[k * SIZE + j] =
              matrix_dist[k * SIZE + i] + matrix_dist[i * SIZE + j];
        }
      }
    }
  }
  }
  }
}

void compareResults(int *B, int *B_GPU) {
  int i, j, fail;
  fail = 0;

  // Compare B and B_GPU
  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < SIZE; i++) {
    {
    int tmc2 = 400 * (28);
    int tm_cost1 = (11 + tmc2);
    #pragma omp task depend(inout: B[0:160401],B_GPU[0:160401]) if(tm_cost1 > 41)
    {
    for (j = 0; j < SIZE; j++) {
      if (percentDiff(B[i * SIZE + j], B_GPU[i * SIZE + j]) > ERROR_THRESHOLD) {
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
  int i;
  int points, var;
  double t_start, t_end;

  int *matrix;
  int *matrix_dist_cpu, *matrix_dist_gpu;

  fprintf(stdout, "<< K-nearest GPU >>\n");

  matrix = (int *)malloc(sizeof(int) * SIZE * SIZE);
  matrix_dist_cpu = (int *)malloc(sizeof(int) * SIZE * SIZE);
  matrix_dist_gpu = (int *)malloc(sizeof(int) * SIZE * SIZE);

  init(matrix, matrix_dist_cpu, matrix_dist_gpu);

  t_start = rtclock();
  Knearest_GPU(matrix, matrix_dist_gpu);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  Knearest_CPU(matrix, matrix_dist_cpu);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(matrix_dist_cpu, matrix_dist_gpu);

  free(matrix);
  free(matrix_dist_cpu);
  free(matrix_dist_gpu);

  return 0;
}

