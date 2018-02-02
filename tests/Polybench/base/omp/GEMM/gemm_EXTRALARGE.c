/**
 * gemm.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
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

/* Problem size */
#ifdef MINI_DATASET
#define NI 25
#define NJ 25
#define NK 25
#endif

#ifdef SMALL_DATASET
#define NI 70
#define NJ 70
#define NK 70
#endif

#ifdef MEDIUM_DATASET
#define NI 256
#define NJ 256
#define NK 256
#endif

#ifdef LARGE_DATASET
#define NI 1024
#define NJ 1024
#define NK 1024
#endif

#ifdef EXTRALARGE_DATASET
#define NI 2300
#define NJ 2300
#define NK 2300
#endif

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0)
 */
#define ALPHA 32412.0f
#define BETA 2123.0f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

#define GPU_DEVICE 1

void gemm(DATA_TYPE *restrict A, DATA_TYPE *restrict B, DATA_TYPE *restrict C) {
  int i, j, k;

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      C[i * NJ + j] *= BETA;

      for (k = 0; k < NK; ++k) {
        C[i * NJ + j] += ALPHA * A[i * NK + k] * B[k * NJ + j];
      }
    }
  }
}

void GPU__gemm(DATA_TYPE *restrict A, DATA_TYPE *restrict B,
               DATA_TYPE *restrict C) {
  int i, j, k;

#pragma omp parallel for collapse(2)
  char RST_AI1 = 0;
  RST_AI1 |= !((A + 0 > B + 5290000)
  || (B + 0 > A + 5290000));
  RST_AI1 |= !((A + 0 > C + 5290000)
  || (C + 0 > A + 5290000));
  RST_AI1 |= !((B + 0 > C + 5290000)
  || (C + 0 > B + 5290000));
  #pragma omp target data map(to: A[0:5290000],B[0:5290000]) map(tofrom: C[0:5290000]) if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      C[i * NJ + j] *= BETA;

      for (k = 0; k < NK; ++k) {
        C[i * NJ + j] += ALPHA * A[i * NK + k] * B[k * NJ + j];
      }
    }
  }
}

void init(DATA_TYPE *restrict A, DATA_TYPE *restrict B, DATA_TYPE *restrict C,
          DATA_TYPE *restrict C_OMP) {
  int i, j;

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NK; j++) {
      A[i * NK + j] = ((DATA_TYPE)i * j) / NI;
    }
  }

  for (i = 0; i < NK; i++) {
    for (j = 0; j < NJ; j++) {
      B[i * NJ + j] = ((DATA_TYPE)i * j + 1) / NJ;
    }
  }

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      C[i * NJ + j] = ((DATA_TYPE)i * j + 2) / NJ;
      C_OMP[i * NJ + j] = ((DATA_TYPE)i * j + 2) / NJ;
    }
  }
}

void compareResults(DATA_TYPE *restrict C,
                    DATA_TYPE *restrict C_outputFromGpu) {
  int i, j, fail;
  fail = 0;

  // Compare C1 and C2
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      if (percentDiff(C[i * NJ + j], C_outputFromGpu[i * NJ + j]) >
          PERCENT_DIFF_ERROR_THRESHOLD) {
        fail++;
      }
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
  DATA_TYPE *C;
  DATA_TYPE *C_outputFromGpu;

  A = (DATA_TYPE *)malloc(NI * NK * sizeof(DATA_TYPE));
  B = (DATA_TYPE *)malloc(NK * NJ * sizeof(DATA_TYPE));
  C = (DATA_TYPE *)malloc(NI * NJ * sizeof(DATA_TYPE));
  C_outputFromGpu = (DATA_TYPE *)malloc(NI * NJ * sizeof(DATA_TYPE));

  fprintf(stdout, "<< Matrix-multiply C=alpha.A.B+beta.C >>\n");

  init(A, B, C, C_outputFromGpu);

  t_start = rtclock();
  GPU__gemm(A, B, C_outputFromGpu);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  gemm(A, B, C);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(C, C_outputFromGpu);

  free(A);
  free(B);
  free(C);
  free(C_outputFromGpu);

  return 0;
}

