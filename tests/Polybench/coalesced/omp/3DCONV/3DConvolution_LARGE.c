/**
 * 3DConvolution.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *	     Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
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
#define ERROR_THRESHOLD 0.5

#define GPU_DEVICE 1

/* Problem size */
#ifdef MINI_DATASET
#define NI 32
#define NJ 32
#define NK 32
#endif

#ifdef SMALL_DATASET
#define NI 64
#define NJ 64
#define NK 64
#endif

#ifdef MEDIUM_DATASET
#define NI 128
#define NJ 128
#define NK 128
#endif

#ifdef LARGE_DATASET
#define NI 256
#define NJ 256
#define NK 256
#endif

#ifdef EXTRALARGE_DATASET
#define NI 512
#define NJ 512
#define NK 512
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void conv3D(DATA_TYPE *restrict A, DATA_TYPE *restrict B) {
  int i, j, k;
  DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

  c11 = +2;
  c21 = +5;
  c31 = -8;
  c12 = -3;
  c22 = +6;
  c32 = -9;
  c13 = +4;
  c23 = +7;
  c33 = +10;

  for (j = 1; j < NJ - 1; ++j) {
    for (i = 1; i < NI - 1; ++i) {
      for (k = 1; k < NK - 1; ++k) {
        B[i * (NK * NJ) + j * NK + k] =
            c11 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c13 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c21 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c23 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c31 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c33 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c12 * A[(i + 0) * (NK * NJ) + (j - 1) * NK + (k + 0)] +
            c22 * A[(i + 0) * (NK * NJ) + (j + 0) * NK + (k + 0)] +
            c32 * A[(i + 0) * (NK * NJ) + (j + 1) * NK + (k + 0)] +
            c11 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] +
            c13 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] +
            c21 * A[(i - 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] +
            c23 * A[(i + 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] +
            c31 * A[(i - 1) * (NK * NJ) + (j + 1) * NK + (k + 1)] +
            c33 * A[(i + 1) * (NK * NJ) + (j + 1) * NK + (k + 1)];
      }
    }
  }
}

void GPU__conv3D(DATA_TYPE *restrict A, DATA_TYPE *restrict B) {
  int i, j, k;
  DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

  c11 = +2;
  c21 = +5;
  c31 = -8;
  c12 = -3;
  c22 = +6;
  c32 = -9;
  c13 = +4;
  c23 = +7;
  c33 = +10;

#pragma omp parallel for
  char RST_AI1 = 0;
  RST_AI1 |= !((A + 0 > B + 16645630)
  || (B + 65793 > A + 16777216));
  #pragma omp target data map(to: A[0:16777216]) map(tofrom: B[65793:16645630]) if(!RST_AI1)
  {
  #pragma omp target if(!RST_AI1)
  for (j = 1; j < NJ - 1; ++j) {
    for (i = 1; i < NI - 1; ++i) {
      int k;
      for (k = 1; k < NK - 1; ++k) {
        B[i * (NK * NJ) + j * NK + k] =
            c11 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c13 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c21 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c23 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c31 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c33 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c12 * A[(i + 0) * (NK * NJ) + (j - 1) * NK + (k + 0)] +
            c22 * A[(i + 0) * (NK * NJ) + (j + 0) * NK + (k + 0)] +
            c32 * A[(i + 0) * (NK * NJ) + (j + 1) * NK + (k + 0)] +
            c11 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] +
            c13 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] +
            c21 * A[(i - 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] +
            c23 * A[(i + 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] +
            c31 * A[(i - 1) * (NK * NJ) + (j + 1) * NK + (k + 1)] +
            c33 * A[(i + 1) * (NK * NJ) + (j + 1) * NK + (k + 1)];
      }
    }
  }
}
}

void init(DATA_TYPE *restrict A) {
  int i, j, k;

  for (i = 0; i < NI; ++i) {
    for (j = 0; j < NJ; ++j) {
      for (k = 0; k < NK; ++k) {
        A[i * (NK * NJ) + j * NK + k] = i % 12 + 2 * (j % 7) + 3 * (k % 13);
      }
    }
  }
}

void compareResults(DATA_TYPE *restrict B, DATA_TYPE *restrict B_GPU) {
  int i, j, k, fail;
  fail = 0;

  // Compare result from cpu and gpu...
  for (i = 1; i < NI - 1; ++i) {
    for (j = 1; j < NJ - 1; ++j) {
      for (k = 1; k < NK - 1; ++k) {
        if (percentDiff(B[i * (NK * NJ) + j * NK + k],
                        B_GPU[i * (NK * NJ) + j * NK + k]) > ERROR_THRESHOLD) {
          fail++;
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
  double t_start, t_end;

  DATA_TYPE *A;
  DATA_TYPE *B;
  DATA_TYPE *B_GPU;

  A = (DATA_TYPE *)malloc(NI * NJ * NK * sizeof(DATA_TYPE));
  B = (DATA_TYPE *)malloc(NI * NJ * NK * sizeof(DATA_TYPE));
  B_GPU = (DATA_TYPE *)malloc(NI * NJ * NK * sizeof(DATA_TYPE));

  fprintf(stdout, ">> Three dimensional (3D) convolution <<\n");

  init(A);

  t_start = rtclock();
  GPU__conv3D(A, B_GPU);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  conv3D(A, B);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(B, B_GPU);

  free(A);
  free(B);
  free(B_GPU);

  return 0;
}

