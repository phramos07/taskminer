/**
 * gramschmidt.c: This file was adapted from PolyBench/GPU 1.0 test
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
#include <math.h>

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
#define M 64
#define N 64
#endif

#ifdef SMALL_DATASET
#define M 128
#define N 128
#endif

#ifdef MEDIUM_DATASET
#define M 256
#define N 256
#endif

#ifdef LARGE_DATASET
#define M 512
#define N 512
#endif

#ifdef EXTRALARGE_DATASET
#define M 1024
#define N 1024
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

#define GPU_DEVICE 1

void gramschmidt(DATA_TYPE *restrict A, DATA_TYPE *restrict R,
                 DATA_TYPE *restrict Q) {
  int i, j, k;
  DATA_TYPE nrm;
  for (k = 0; k < N; k++) {
    nrm = 0;
    for (i = 0; i < M; i++) {
      nrm += A[i * N + k] * A[i * N + k];
    }

    R[k * N + k] = sqrt(nrm);
    for (i = 0; i < M; i++) {
      Q[i * N + k] = A[i * N + k] / R[k * N + k];
    }

    for (j = k + 1; j < N; j++) {
      R[k * N + j] = 0;
      for (i = 0; i < M; i++) {
        R[k * N + j] += Q[i * N + k] * A[i * N + j];
      }
      for (i = 0; i < M; i++) {
        A[i * N + j] = A[i * N + j] - Q[i * N + k] * R[k * N + j];
      }
    }
  }
}

void GPU__gramschmidt(DATA_TYPE *restrict A, DATA_TYPE *restrict R,
                      DATA_TYPE *restrict Q) {
  int i, j, k;
  DATA_TYPE nrm;

  char RST_AI1 = 0;
  RST_AI1 |= !((A + 0 > Q + 65536)
  || (Q + 0 > A + 65536));
  RST_AI1 |= !((A + 0 > R + 65536)
  || (R + 0 > A + 65536));
  RST_AI1 |= !((Q + 0 > R + 65536)
  || (R + 0 > Q + 65536));
  #pragma acc data pcopy(A[0:65536],Q[0:65536],R[0:65536]) if(!RST_AI1)
  #pragma acc kernels if(!RST_AI1)
  for (k = 0; k < N; k++) {
    // CPU
    nrm = 0;
    for (i = 0; i < M; i++) {
      nrm += A[i * N + k] * A[i * N + k];
    }
    R[k * N + k] = sqrt(nrm);

    #pragma acc loop independent
    for (i = 0; i < M; i++) {
      Q[i * N + k] = A[i * N + k] / R[k * N + k];
    }

    #pragma acc loop independent
    for (j = k + 1; j < N; j++) {
      R[k * N + j] = 0;
      for (i = 0; i < M; i++) {
        R[k * N + j] += Q[i * N + k] * A[i * N + j];
      }
      for (i = 0; i < M; i++) {
        A[i * N + j] = A[i * N + j] - Q[i * N + k] * R[k * N + j];
      }
    }
  }
}

void init_array(DATA_TYPE *restrict A, DATA_TYPE *restrict A2) {
  int i, j;

  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      A[i * N + j] = ((DATA_TYPE)(i + 1) * (j + 1)) / (M + 1);
      A2[i * N + j] = A[i * N + j];
    }
  }
}

void compareResults(DATA_TYPE *restrict A,
                    DATA_TYPE *restrict A_outputFromGpu) {
  int i, j, fail;
  fail = 0;

  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      if (percentDiff(A[i * N + j], A_outputFromGpu[i * N + j]) >
          PERCENT_DIFF_ERROR_THRESHOLD) {
        fail++;
        // printf("i: %d j: %d \n1: %f\n 2: %f\n", i, j, A[i*N + j],
        // A_outputFromGpu[i*N + j]);
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
  DATA_TYPE *A_outputFromGpu;
  DATA_TYPE *R;
  DATA_TYPE *Q;

  A = (DATA_TYPE *)malloc(M * N * sizeof(DATA_TYPE));
  A_outputFromGpu = (DATA_TYPE *)malloc(M * N * sizeof(DATA_TYPE));
  R = (DATA_TYPE *)malloc(M * N * sizeof(DATA_TYPE));
  Q = (DATA_TYPE *)malloc(M * N * sizeof(DATA_TYPE));

  fprintf(stdout, "<< Gram-Schmidt decomposition >>\n");

  init_array(A, A_outputFromGpu);

  t_start = rtclock();
  GPU__gramschmidt(A_outputFromGpu, R, Q);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  gramschmidt(A, R, Q);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(A, A_outputFromGpu);

  free(A);
  free(A_outputFromGpu);
  free(R);
  free(Q);

  return 0;
}

