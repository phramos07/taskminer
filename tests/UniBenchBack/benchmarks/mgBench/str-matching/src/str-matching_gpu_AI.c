#include <omp.h>
#ifndef taskminerutils
#define taskminerutils
static int taskminer_depth_cutoff = 0;
#define DEPTH_CUTOFF omp_get_num_threads()
char cutoff_test = 0;
#endif
/*
   This program performs string matching on the GPU with
   dynamically allocated vector.

    Author: Gleison Souza Diniz Mendonça
    Date: 04-01-2015
    version 2.0

    Run:
    ipmacc string-matching _gpu.c -o str
    ./str matrix-size
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include "../../common/mgbenchUtilFunctions.h"

#define SIZE 1000
#define SIZE2 500
#define GPU_DEVICE 0
#define PERCENT_DIFF_ERROR_THRESHOLD 0.01

/// initialize the two strings
void init(char *frase, char *palavra) {
  int i;
  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < SIZE; i++) {
    {
    int tm_cost2 = (10);
    #pragma omp task depend(inout: frase[0:1001]) if(tm_cost2 > 41)
    {
    frase[i] = 'a';
  }
  }
  }

  frase[i] = '\0';
  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < SIZE2; i++) {
    {
    int tm_cost1 = (10);
    #pragma omp task depend(inout: palavra[0:501]) if(tm_cost1 > 41)
    {
    palavra[i] = 'a';
  }
  }
  }

  palavra[i] = '\0';
}

/// string matching algorithm GPU
/// s = size of longer string
/// p = size of less string
int string_matching_GPU(char *frase, char *palavra) {
  int i, diff, j, parallel_size, count = 0;
  diff = SIZE - SIZE2;

  parallel_size = 10000;
  int *vector;
  vector = (int *)malloc(sizeof(int) * parallel_size);

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < parallel_size; i++) {
    {
    int tm_cost4 = (10);
    #pragma omp task depend(inout: vector[0:10000]) if(tm_cost4 > 41)
    {
    vector[i] = 0;
  }
  }
  }

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < diff; i++) {
    {
    int tmc3 = 500 * (22);
    int tm_cost2 = (26 + tmc3);
    #pragma omp task depend(inout: frase[0:1001],palavra[0:501],vector[0:10000]) if(tm_cost2 > 41)
    {
    {
    int tmc3 = 500 * (22);
    int tm_cost2 = (32 + tmc3);
    #pragma omp task depend(inout: frase[0:1001],palavra[0:501],vector[0:10000]) if(tm_cost2 > 41)
    {
    int v;
    v = 0;
    for (j = 0; j < SIZE2; j++) {
      if (frase[(i + j)] != palavra[j]) {
        v = 1;
      }
    }
    if (v == 0) {
      vector[i % parallel_size]++;
    }
  }
  }
  }

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < parallel_size; i++) {
    {
    int tm_cost1 = (13);
    #pragma omp task depend(inout: vector[0:10000]) if(tm_cost1 > 41)
    {
    count += vector[i];
  }
  }
  }

  return count;
}

int string_matching_CPU(char *frase, char *palavra) {
  int i, j, diff, count;
  diff = SIZE - SIZE2;
  count = 0;

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < diff; i++) {
    {
    int tmc2 = 500 * (22);
    int tm_cost1 = (18 + tmc2);
    #pragma omp task depend(inout: frase[0:1001],palavra[0:501]) if(tm_cost1 > 41)
    {
    int v;
    v = 0;
    for (j = 0; j < SIZE2; j++) {
      if (frase[(i + j)] != palavra[j]) {
        v = 1;
      }
    }
    if (v == 0) {
      count++;
    }
  }
  }
  }

  return count;
}

int main(int argc, char *argv[]) {
  double t_start, t_end;
  char *frase;
  char *palavra;

  int count_cpu, count_gpu;

  frase = (char *)malloc(sizeof(char) * (SIZE + 1));
  palavra = (char *)malloc(sizeof(char) * (SIZE2 + 1));

  init(frase, palavra);

  fprintf(stdout, "<< String Matching >>\n");

  t_start = rtclock();
  count_cpu = string_matching_CPU(frase, palavra);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  count_gpu = string_matching_GPU(frase, palavra);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  if (count_cpu == count_gpu)
    printf("Corrects answers: %d = %d\n", count_cpu, count_gpu);
  else
    printf("Error: %d != %d\n", count_cpu, count_gpu);

  free(frase);
  free(palavra);

  return 0;
}

