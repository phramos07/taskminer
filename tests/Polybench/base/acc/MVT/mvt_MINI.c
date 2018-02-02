/**
 * mvt.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *           Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
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
#define N 512
#endif

#ifdef LARGE_DATASET
#define N 2048
#endif

#ifdef EXTRALARGE_DATASET
#define N 4096
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE *restrict A, DATA_TYPE *restrict x1,
                DATA_TYPE *restrict x2, DATA_TYPE *restrict y1,
                DATA_TYPE *restrict y2, DATA_TYPE *restrict x1_gpu,
                DATA_TYPE *restrict x2_gpu) {
  int i, j;

  for (i = 0; i < N; i++) {
    x1[i] = ((DATA_TYPE)i) / N;
    x2[i] = ((DATA_TYPE)i + 1) / N;
    x1_gpu[i] = x1[i];
    x2_gpu[i] = x2[i];
    y1[i] = ((DATA_TYPE)i + 3) / N;
    y2[i] = ((DATA_TYPE)i + 4) / N;
    for (j = 0; j < N; j++) {
      A[i * N + j] = ((DATA_TYPE)i * j) / N;
    }
  }
}

void runMvt(DATA_TYPE *restrict a, DATA_TYPE *restrict x1,
            DATA_TYPE *restrict x2, DATA_TYPE *restrict y1,
            DATA_TYPE *restrict y2) {
  int i, j;

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      x1[i] = x1[i] + a[i * N + j] * y1[j];
    }
  }

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      x2[i] = x2[i] + a[j * N + i] * y2[j];
    }
  }
}

void GPU__runMvt(DATA_TYPE *restrict a, DATA_TYPE *restrict x1,
                 DATA_TYPE *restrict x2, DATA_TYPE *restrict y1,
                 DATA_TYPE *restrict y2) {
  int i;

// Note that you must collapse only outer loop to avoid conflicts
  char RST_AI1 = 0;
  RST_AI1 |= !((a + 0 > x1 + 128)
  || (x1 + 0 > a + 16384));
  RST_AI1 |= !((a + 0 > y1 + 128)
  || (y1 + 0 > a + 16384));
  RST_AI1 |= !((x1 + 0 > y1 + 128)
  || (y1 + 0 > x1 + 128));
  #pragma acc data pcopyin(a[0:16384],y1[0:128]) pcopy(x1[0:128]) if(!RST_AI1)
  #pragma acc kernels if(!RST_AI1)
  #pragma acc loop independent
  for (i = 0; i < N; i++) {
    int j;
    for (j = 0; j < N; j++) {
      x1[i] = x1[i] + a[i * N + j] * y1[j];
    }
  }

  char RST_AI2 = 0;
  RST_AI2 |= !((a + 0 > x2 + 128)
  || (x2 + 0 > a + 16384));
  RST_AI2 |= !((a + 0 > y2 + 128)
  || (y2 + 0 > a + 16384));
  RST_AI2 |= !((x2 + 0 > y2 + 128)
  || (y2 + 0 > x2 + 128));
  #pragma acc data pcopyin(a[0:16384],y2[0:128]) pcopy(x2[0:128]) if(!RST_AI2)
  #pragma acc kernels if(!RST_AI2)
  #pragma acc loop independent
  for (i = 0; i < N; i++) {
    int j;
    for (j = 0; j < N; j++) {
      x2[i] = x2[i] + a[j * N + i] * y2[j];
    }
  }
}

void compareResults(DATA_TYPE *restrict x1,
                    DATA_TYPE *restrict x1_outputFromGpu,
                    DATA_TYPE *restrict x2,
                    DATA_TYPE *restrict x2_outputFromGpu) {
  int i, fail;
  fail = 0;

  for (i = 0; i < N; i++) {
    if (percentDiff(x1[i], x1_outputFromGpu[i]) >
        PERCENT_DIFF_ERROR_THRESHOLD) {
      fail++;
    }

    if (percentDiff(x2[i], x2_outputFromGpu[i]) >
        PERCENT_DIFF_ERROR_THRESHOLD) {
      fail++;
    }
  }

  // Print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
         "Percent: %d\n",
         PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

int main() {
  double t_start, t_end;

  DATA_TYPE *a;
  DATA_TYPE *x1;
  DATA_TYPE *x2;
  DATA_TYPE *x1_outputFromGpu;
  DATA_TYPE *x2_outputFromGpu;
  DATA_TYPE *y_1;
  DATA_TYPE *y_2;

  a = (DATA_TYPE *)malloc(N * N * sizeof(DATA_TYPE));
  x1 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  x2 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  x1_outputFromGpu = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  x2_outputFromGpu = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  y_1 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  y_2 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));

  fprintf(stdout, "<< Matrix Vector Product and Transpose >>\n");

  init_array(a, x1, x2, y_1, y_2, x1_outputFromGpu, x2_outputFromGpu);

  t_start = rtclock();
  GPU__runMvt(a, x1_outputFromGpu, x2_outputFromGpu, y_1, y_2);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  // run the algorithm on the CPU
  runMvt(a, x1, x2, y_1, y_2);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(x1, x1_outputFromGpu, x2, x2_outputFromGpu);

  free(a);
  free(x1);
  free(x2);
  free(x1_outputFromGpu);
  free(x2_outputFromGpu);
  free(y_1);
  free(y_2);

  return 0;
}

