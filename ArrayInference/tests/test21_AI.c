#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define SIZE 1000
#define GPU_DEVICE 0
#define PERCENT_DIFF_ERROR_THRESHOLD 0.01

// Initialize matrices.
void init(int *a, int *b, int *c_cpu, int *c_gpu) {
  int i, j;
  char RST_AI1 = 0;
  RST_AI1 |= !((a + 0 > b + 1000000)
  || (b + 0 > a + 1000000));
  RST_AI1 |= !((a + 0 > c_cpu + 1000000)
  || (c_cpu + 0 > a + 1000000));
  RST_AI1 |= !((a + 0 > c_gpu + 1000000)
  || (c_gpu + 0 > a + 1000000));
  RST_AI1 |= !((b + 0 > c_cpu + 1000000)
  || (c_cpu + 0 > b + 1000000));
  RST_AI1 |= !((b + 0 > c_gpu + 1000000)
  || (c_gpu + 0 > b + 1000000));
  RST_AI1 |= !((c_cpu + 0 > c_gpu + 1000000)
  || (c_gpu + 0 > c_cpu + 1000000));
  #pragma acc data pcopyout(a[0:1000000],b[0:1000000],c_cpu[0:1000000],c_gpu[0:1000000]) if(!RST_AI1)
  #pragma acc kernels if(!RST_AI1)
  #pragma acc loop independent 
  for (i = 0; i < SIZE; ++i) {
    #pragma acc loop independent 
    for (j = 0; j < SIZE; ++j) {
      a[i * SIZE + j] = (i + j) % 100;
      b[i * SIZE + j] = (i + j) % 100;
      c_cpu[i * SIZE + j] = 0;
      c_gpu[i * SIZE + j] = 0;
    }
  }
}

/// matrix multiplication algorithm GPU
/// s = size of matrix
void mul_GPU(int *a, int *b, int *c) {
  int i, j, k;
  int sum = 0.0;

  char RST_AI1 = 0;
  RST_AI1 |= !((a + 0 > b + 1000000)
  || (b + 0 > a + 1000000));
  RST_AI1 |= !((a + 0 > c + 1000000)
  || (c + 0 > a + 1000000));
  RST_AI1 |= !((b + 0 > c + 1000000)
  || (c + 0 > b + 1000000));
  #pragma acc data pcopyin(a[0:1000000],b[0:1000000]) pcopyout(c[0:1000000]) if(!RST_AI1)
  #pragma acc kernels if(!RST_AI1)
  #pragma acc loop independent 
  for (i = 0; i < SIZE; ++i) {
    #pragma acc loop independent 
    for (j = 0; j < SIZE; ++j) {
      sum = 0.0;
      for (k = 0; k < SIZE; ++k) {
        sum = sum + a[i * SIZE + k] * b[k * SIZE + j];
      }
      c[i * SIZE + j] = sum;
    }
  }
}

void mul_CPU(int *a, int *b, int *c) {
  int i, j, k;
  int sum = 0.0;
  char RST_AI1 = 0;
  RST_AI1 |= !((a + 0 > b + 1000000)
  || (b + 0 > a + 1000000));
  RST_AI1 |= !((a + 0 > c + 1000000)
  || (c + 0 > a + 1000000));
  RST_AI1 |= !((b + 0 > c + 1000000)
  || (c + 0 > b + 1000000));
  #pragma acc data pcopyin(a[0:1000000],b[0:1000000]) pcopyout(c[0:1000000]) if(!RST_AI1)
  #pragma acc kernels if(!RST_AI1)
  #pragma acc loop independent 
  for (i = 0; i < SIZE; ++i) {
    #pragma acc loop independent 
    for (j = 0; j < SIZE; ++j) {
      sum = 0.0;
      for (k = 0; k < SIZE; ++k) {
        sum = sum + a[i * SIZE + k] * b[k * SIZE + j];
      }
      c[i * SIZE + j] = sum;
    }
  }
}

void compareResults(int *b_cpu, int *b_gpu) {
  int i, j, fail;
  fail = 0;

  for (i = 0; i < SIZE; i++) {
    for (j = 0; j < SIZE; j++) {
      if (b_cpu[i * SIZE + j] != b_gpu[i * SIZE + j]) {
        fail++;
      }
    }
  }

  // Print results
  printf("Different values: %d\n", fail);
}

int main(int argc, char *argv[]) {

  clock_t t_start, t_end;
  double time = 0.0;
  int *a, *b, *c_cpu, *c_gpu;

  a = (int *)malloc(sizeof(int) * SIZE * SIZE);
  b = (int *)malloc(sizeof(int) * SIZE * SIZE);
  c_cpu = (int *)malloc(sizeof(int) * SIZE * SIZE);
  c_gpu = (int *)malloc(sizeof(int) * SIZE * SIZE);

  init(a, b, c_cpu, c_gpu);

  fprintf(stdout, "<< Matrix Multiplication >>\n");

  t_start = clock();
  mul_GPU(a, b, c_gpu);
  t_end = clock();
  time = (double)(t_end - t_start) / CLOCKS_PER_SEC;
  fprintf(stdout, "GPU Runtime: %lf seconds\n", time);

  t_start = clock();
  mul_CPU(a, b, c_cpu);
  t_end = clock();
  time = (double)(t_end - t_start) / CLOCKS_PER_SEC;
  fprintf(stdout, "CPU Runtime: %lf seconds\n", time);

  compareResults(c_cpu, c_gpu);

  free(a);
  free(b);
  free(c_cpu);
  free(c_gpu);

  return 0;
}

