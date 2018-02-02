/**
 * syr2k.c: This file was adapted from PolyBench/GPU 1.0 test suite
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

/* Problem size */
#ifdef MINI_DATASET
#define N 64
#define M 64
#endif

#ifdef SMALL_DATASET
#define N 128
#define M 128
#endif

#ifdef MEDIUM_DATASET
#define N 256
#define M 256
#endif

#ifdef LARGE_DATASET
#define N 1200
#define M 1200
#endif

#ifdef EXTRALARGE_DATASET
#define N 2400
#define M 2400
#endif

#define GPU_DEVICE 1

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0)
 */
#define ALPHA 12435
#define BETA 4546

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_arrays(DATA_TYPE * A, DATA_TYPE * B,
                 DATA_TYPE * C, DATA_TYPE * C_GPU) {
  int i, j;

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      C[i * N + j] = ((DATA_TYPE)i * j + 2) / N;
      C_GPU[i * N + j] = C[i * N + j];
    }

    for (j = 0; j < M; j++) {
      A[i * N + j] = ((DATA_TYPE)i * j) / N;
      B[i * N + j] = ((DATA_TYPE)i * j + 1) / N;
    }
  }
}

void syr2k(DATA_TYPE * A, DATA_TYPE * B,
           DATA_TYPE * C) {
  int i, j, k;

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      C[i * N + j] *= BETA;
    }
  }

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      for (k = 0; k < M; k++) {
        C[i * N + j] += ALPHA * A[i * M + k] * B[j * M + k];
        C[i * N + j] += ALPHA * B[i * M + k] * A[j * M + k];
      }
    }
  }
}

void GPU__syr2k(DATA_TYPE * A, DATA_TYPE * B,
                DATA_TYPE * C) {
  int i, j, k;

  char RST_AI1 = 0;
  RST_AI1 |= !((A + 0 > B + 5760000)
  || (B + 0 > A + 5760000));
  RST_AI1 |= !((A + 0 > C + 5760000)
  || (C + 0 > A + 5760000));
  RST_AI1 |= !((B + 0 > C + 5760000)
  || (C + 0 > B + 5760000));
  #pragma acc data pcopyin(A[0:5760000],B[0:5760000]) pcopy(C[0:5760000]) if(!RST_AI1)
  #pragma acc kernels if(!RST_AI1)
  #pragma acc loop independent
  for (i = 0; i < N; i++) {
    #pragma acc loop independent
    for (j = 0; j < N; j++) {
      C[i * N + j] *= BETA;
      for (k = 0; k < M; k++) {
        C[i * N + j] += ALPHA * A[i * M + k] * B[j * M + k] +
                        ALPHA * B[i * M + k] * A[j * M + k];
      }
    }
  }
}

void compareResults(DATA_TYPE * C,
                    DATA_TYPE * C_outputFromGpu) {
  int i, j, fail;
  fail = 0;

  // Compare C with D
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      if (percentDiff(C[i * N + j], C_outputFromGpu[i * N + j]) >
          PERCENT_DIFF_ERROR_THRESHOLD) {
        fail++;
      }
    }
  }

  // print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
         "Percent: %d\n",
         PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

int main() {
  double t_start, t_end;

  DATA_TYPE *A;
  DATA_TYPE *B;
  DATA_TYPE *C;
  DATA_TYPE *C_outputFromGpu;

  A = (DATA_TYPE *)malloc(N * M * sizeof(DATA_TYPE));
  B = (DATA_TYPE *)malloc(N * M * sizeof(DATA_TYPE));
  C = (DATA_TYPE *)malloc(N * M * sizeof(DATA_TYPE));
  C_outputFromGpu = (DATA_TYPE *)malloc(N * M * sizeof(DATA_TYPE));

  fprintf(stdout, "<< Symmetric rank-2k operations >>\n");

  init_arrays(A, B, C, C_outputFromGpu);

  t_start = rtclock();
  GPU__syr2k(A, B, C_outputFromGpu);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  syr2k(A, B, C);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(C, C_outputFromGpu);

  free(A);
  free(B);
  free(C);
  free(C_outputFromGpu);

  return 0;
}

