/**
 * gesummv.c: This file was adapted from PolyBench/GPU 1.0 test
 * suite to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
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
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 1

/* Problem size */
#ifdef MINI_DATASET
#define N 128
#endif

#ifdef SMALL_DATASET
#define N 256
#endif

#ifdef MEDIUM_DATASET
#define N 1024
#endif

#ifdef LARGE_DATASET
#define N 2048
#endif

#ifdef EXTRALARGE_DATASET
#define N 4096
#endif

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0)
 */
#define ALPHA 43532.0f
#define BETA 12313.0f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void gesummv(DATA_TYPE *restrict A, DATA_TYPE *restrict B,
             DATA_TYPE *restrict x, DATA_TYPE *restrict y,
             DATA_TYPE *restrict tmp) {
  int i, j;

  for (i = 0; i < N; i++) {
    tmp[i] = 0;
    y[i] = 0;
    for (j = 0; j < N; j++) {
      tmp[i] = A[i * N + j] * x[j] + tmp[i];
      y[i] = B[i * N + j] * x[j] + y[i];
    }

    y[i] = ALPHA * tmp[i] + BETA * y[i];
  }
}

void GPU__gesummv(DATA_TYPE *restrict A, DATA_TYPE *restrict B,
                  DATA_TYPE *restrict x, DATA_TYPE *restrict y,
                  DATA_TYPE *restrict tmp) {
  int i, j;

  char RST_AI1 = 0;
  RST_AI1 |= !((A + 0 > B + 1048576)
  || (B + 0 > A + 1048576));
  RST_AI1 |= !((A + 0 > tmp + 1024)
  || (tmp + 0 > A + 1048576));
  RST_AI1 |= !((A + 0 > x + 1024)
  || (x + 0 > A + 1048576));
  RST_AI1 |= !((A + 0 > y + 1024)
  || (y + 0 > A + 1048576));
  RST_AI1 |= !((B + 0 > tmp + 1024)
  || (tmp + 0 > B + 1048576));
  RST_AI1 |= !((B + 0 > x + 1024)
  || (x + 0 > B + 1048576));
  RST_AI1 |= !((B + 0 > y + 1024)
  || (y + 0 > B + 1048576));
  RST_AI1 |= !((tmp + 0 > x + 1024)
  || (x + 0 > tmp + 1024));
  RST_AI1 |= !((tmp + 0 > y + 1024)
  || (y + 0 > tmp + 1024));
  RST_AI1 |= !((x + 0 > y + 1024)
  || (y + 0 > x + 1024));
  #pragma acc data pcopyin(A[0:1048576],B[0:1048576],x[0:1024]) pcopy(tmp[0:1024],y[0:1024]) if(!RST_AI1)
  {
  #pragma acc kernels if(!RST_AI1)
  #pragma acc loop independent
  for (i = 0; i < N; i++) {
    tmp[i] = 0;
    y[i] = 0;
    for (j = 0; j < N; j++) {
      tmp[i] = A[i * N + j] * x[j] + tmp[i];
      y[i] = B[i * N + j] * x[j] + y[i];
    }

    y[i] = ALPHA * tmp[i] + BETA * y[i];
  }
}
}

void init(DATA_TYPE *restrict A, DATA_TYPE *restrict x) {
  int i, j;

  for (i = 0; i < N; i++) {
    x[i] = ((DATA_TYPE)i) / N;

    for (j = 0; j < N; j++) {
      A[i * N + j] = ((DATA_TYPE)i * j) / N;
    }
  }
}

void compareResults(DATA_TYPE *restrict y,
                    DATA_TYPE *restrict y_outputFromGpu) {
  int i, fail;
  fail = 0;

  for (i = 0; i < (N); i++) {
    if (percentDiff(y[i], y_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
      fail++;
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
  DATA_TYPE *x;
  DATA_TYPE *y;
  DATA_TYPE *y_outputFromGpu;
  DATA_TYPE *tmp;

  A = (DATA_TYPE *)malloc(N * N * sizeof(DATA_TYPE));
  B = (DATA_TYPE *)malloc(N * N * sizeof(DATA_TYPE));
  x = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  y = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  y_outputFromGpu = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  tmp = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));

  fprintf(stdout, "<< Scalar, Vector and Matrix Multiplication >>\n");

  init(A, x);

  t_start = rtclock();
  GPU__gesummv(A, B, x, y_outputFromGpu, tmp);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  //gesummv(A, B, x, y, tmp);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(y, y_outputFromGpu);

  free(A);
  free(B);
  free(x);
  free(y);
  free(y_outputFromGpu);
  free(tmp);

  return 0;
}

