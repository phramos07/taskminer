#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <openacc.h>
#define IPMACC_MAX1(A)   (A)
#define IPMACC_MAX2(A,B) (A>B?A:B)
#define IPMACC_MAX3(A,B,C) (A>B?(A>C?A:(B>C?B:C)):(B>C?C:B))
#ifdef __cplusplus
#include "openacc_container.h"
#endif

#include <cuda.h>



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include "../../common/polybenchUtilFuncts.h"


#define ERROR_THRESHOLD 0.05
#define GPU_DEVICE 1


#define N 1024
#define M 1024



#define alpha 12435
#define beta 4546


typedef float DATA_TYPE;

DATA_TYPE A [N] [M];
DATA_TYPE C [N] [M];
DATA_TYPE D [N] [M];

void init_arrays()
{
  int i, j;

  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      A [i] [j] = ((DATA_TYPE)i * j) / N;
    }
    for (j = 0; j < M; j++) {
      C [i] [j] = ((DATA_TYPE)i * j + 2) / N;
      D [i] [j] = C [i] [j];
    }
  }
}

void syrk()
{
  int i, j, k;

  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      C [i] [j] *= beta;
    }
  }

  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      for (k = 0; k < M; k++) {
        C [i] [j] += alpha * A [i] [k] * A [j] [k];
      }
    }
  }
}

void compareResults()
{
  int i, j, fail;
  fail = 0;

  
  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      if (percentDiff(C [i] [j], D [i] [j]) > ERROR_THRESHOLD) {
        fail++;
      }
    }
  }

  
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", ERROR_THRESHOLD, fail);
}

void syrkGPU()
{
  int i, j;
  double t_start, t_end;

  t_start = rtclock();

#pragma omp target map(to: A) map(tofrom: D) device (GPU_DEVICE)
#pragma omp parallel for collapse(2)
  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      D [i] [j] *= beta;
      int k;
      for (k = 0; k < M; k++) {
        D [i] [j] += alpha * A [i] [k] * A [j] [k];
      }
    }
  }

  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
}

int main()
{
  double t_start, t_end;

  init_arrays();
  syrkGPU();
  t_start = rtclock();
  syrk();
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
  compareResults();
  return 0;
}



