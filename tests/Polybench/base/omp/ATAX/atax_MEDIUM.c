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

#define SMALL_FLOAT_VAL 0.00000001f

double rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0)
    printf("Error return from gettimeofday: %d", stat);
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

float absVal(float a) {
  if (a < 0) {
    return (a * -1);
  } else {
    return a;
  }
}

float percentDiff(double val1, double val2) {
  double val3 = (val1 >= val2) ? (val1 - val2) : (val2 - val1);
  if (val3 < 0.5) {
    return 0.0f;
  }

  else {
    return 100.0f *
           (absVal(absVal(val1 - val2) / absVal(val1 + SMALL_FLOAT_VAL)));
  }
}

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

/* Problem size. */
#ifdef MINI_DATASET
#define NX 256
#define NY 256
#endif

#ifdef SMALL_DATASET
#define NX 512
#define NY 512
#endif

#ifdef MEDIUM_DATASET
#define NX 1024
#define NY 1024
#endif

#ifdef LARGE_DATASET
#define NX 4096
#define NY 4096
#endif

#ifdef EXTRALARGE_DATASET
#define NX 8192
#define NY 8192
#endif

#define GPU_DEVICE 1

#ifndef M_PI
#define M_PI 3.14159
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE *restrict x, DATA_TYPE *restrict A) {
  int i, j;

  for (i = 0; i < NX; i++) {
    x[i] = i * M_PI;
    for (j = 0; j < NY; j++) {
      A[i * NY + j] = ((DATA_TYPE)i * (j)) / NX;
    }
  }
}

void compareResults(DATA_TYPE *restrict z,
                    DATA_TYPE *restrict z_outputFromGpu) {
  int i, fail;
  fail = 0;

  for (i = 0; i < NY; i++) {
    if (percentDiff(z[i], z_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
      fail++;
    }
  }

  // print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
         "Percent: %d\n",
         PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void atax_cpu(DATA_TYPE *restrict A, DATA_TYPE *restrict x,
              DATA_TYPE *restrict y, DATA_TYPE *restrict tmp) {
  int i, j;

  for (i = 0; i < NY; i++) {
    y[i] = 0;
  }

  for (i = 0; i < NX; i++) {
    tmp[i] = 0;

    for (j = 0; j < NY; j++) {
      tmp[i] = tmp[i] + A[i * NY + j] * x[j];
    }

    for (j = 0; j < NY; j++) {
      y[j] = y[j] + A[i * NY + j] * tmp[i];
    }
  }
}

void GPU__atax(DATA_TYPE *restrict A, DATA_TYPE *restrict x,
               DATA_TYPE *restrict y, DATA_TYPE *restrict tmp) {
  int i, j;

  #pragma omp target data map(tofrom: y[0:1024])
  #pragma omp target
  for (i = 0; i < NY; i++) {
    y[i] = 0;
  }

#pragma omp parallel for
  char RST_AI2 = 0;
  RST_AI2 |= !((A + 0 > tmp + 1024)
  || (tmp + 0 > A + 1048576));
  RST_AI2 |= !((A + 0 > x + 1024)
  || (x + 0 > A + 1048576));
  RST_AI2 |= !((tmp + 0 > x + 1024)
  || (x + 0 > tmp + 1024));
  #pragma omp target data map(to: A[0:1048576],x[0:1024]) map(tofrom: tmp[0:1024]) if(!RST_AI2)
  #pragma omp target if(!RST_AI2)
  for (i = 0; i < NX; i++) {
    tmp[i] = 0;
    int j;
    for (j = 0; j < NY; j++) {
      tmp[i] = tmp[i] + A[i * NY + j] * x[j];
    }
  }

// Note that the Loop has been reversed
#pragma omp parallel for collapse(1)
  char RST_AI3 = 0;
  RST_AI3 |= !((A + 0 > tmp + 1024)
  || (tmp + 0 > A + 1048576));
  RST_AI3 |= !((A + 0 > y + 1024)
  || (y + 0 > A + 1048576));
  RST_AI3 |= !((tmp + 0 > y + 1024)
  || (y + 0 > tmp + 1024));
  #pragma omp target data map(to: A[0:1048576],tmp[0:1024]) map(tofrom: y[0:1024]) if(!RST_AI3)
  #pragma omp target if(!RST_AI3)
  for (j = 0; j < NY; j++)
    for (i = 0; i < NX; i++) {
      { y[j] = y[j] + A[i * NY + j] * tmp[i]; }
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
  GPU__atax(A, x, y_outputFromGpu, tmp);
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

