/**
 * covariance.c: This file was adapted from PolyBench/GPU 1.0 test
 * suite to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
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
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

#define GPU_DEVICE 1

/* Problem size */
#ifdef MINI_DATASET
#define M 128
#define N 128
#endif

#ifdef SMALL_DATASET
#define M 256
#define N 256
#endif

#ifdef MEDIUM_DATASET
#define M 512
#define N 512
#endif

#ifdef LARGE_DATASET
#define M 1024
#define N 1024
#endif

#ifdef EXTRALARGE_DATASET
#define M 2048
#define N 2048
#endif

#define sqrt_of_array_cell(x, j) sqrt(x[j])

#define FLOAT_N 3214212.01
#define EPS 0.005

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_arrays(DATA_TYPE *restrict data) {
  int i, j;

  for (i = 1; i < (M + 1); i++) {
    for (j = 1; j < (N + 1); j++) {
      data[i * (N + 1) + j] = ((DATA_TYPE)i * j) / M;
    }
  }
}

void compareResults(DATA_TYPE *restrict symmat,
                    DATA_TYPE *restrict symmat_outputFromGpu) {
  int i, j, fail;
  fail = 0;

  for (i = 1; i < (M + 1); i++) {
    for (j = 1; j < (N + 1); j++) {
      if (percentDiff(symmat[i * (N + 1) + j],
                      symmat_outputFromGpu[i * (N + 1) + j]) >
          PERCENT_DIFF_ERROR_THRESHOLD) {
        fail++;
      }
    }
  }
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
         "Percent: %d\n",
         PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void covariance(DATA_TYPE *restrict data, DATA_TYPE *restrict symmat,
                DATA_TYPE *restrict mean) {
  int i, j, j1, j2;

  /* Determine mean of column vectors of input data matrix */
  for (j = 1; j < (M + 1); j++) {
    mean[j] = 0.0;
    for (i = 1; i < (N + 1); i++) {
      mean[j] += data[i * (M + 1) + j];
    }
    mean[j] /= FLOAT_N;
  }

  /* Center the column vectors. */
  for (i = 1; i < (N + 1); i++) {
    for (j = 1; j < (M + 1); j++) {
      data[i * (M + 1) + j] -= mean[j];
    }
  }

  /* Calculate the m * m covariance matrix. */
  for (j1 = 1; j1 < (M + 1); j1++) {
    for (j2 = j1; j2 < (M + 1); j2++) {
      symmat[j1 * (M + 1) + j2] = 0.0;
      for (i = 1; i < N + 1; i++) {
        symmat[j1 * (M + 1) + j2] +=
            data[i * (M + 1) + j1] * data[i * (M + 1) + j2];
      }
      symmat[j2 * (M + 1) + j1] = symmat[j1 * (M + 1) + j2];
    }
  }
}

void GPU__covariance(DATA_TYPE *restrict data, DATA_TYPE *restrict symmat,
                     DATA_TYPE *restrict mean) {
  int i, j, j1, j2;

/* Determine mean of column vectors of input data matrix */
  #pragma acc kernels
  #pragma acc loop independent
  for (j = 1; j < (M + 1); j++) {
    mean[j] = 0.0;
    for (i = 1; i < (N + 1); i++) {
      mean[j] += data[i * (M + 1) + j];
    }
    mean[j] /= FLOAT_N;
  }

/* Center the column vectors. */
  #pragma acc kernels
  #pragma acc loop independent
  for (i = 1; i < (N + 1); i++)
  {
    #pragma acc loop independent
    for (j = 1; j < (M + 1); j++) {
      data[i * (M + 1) + j] -= mean[j];
    }
  }

/* Calculate the m * m covariance matrix. */
  #pragma acc kernels
  #pragma acc loop independent
  for (j1 = 1; j1 < (M + 1); j1++) {
    #pragma acc loop independent
    for (j2 = j1; j2 < (M + 1); j2++) {
      symmat[j1 * (M + 1) + j2] = 0.0;
      for (i = 1; i < N + 1; i++) {
        symmat[j1 * (M + 1) + j2] +=
            data[i * (M + 1) + j1] * data[i * (M + 1) + j2];
      }
      symmat[j2 * (M + 1) + j1] = symmat[j1 * (M + 1) + j2];
    }
  }
}

int main() {
  double t_start, t_end;

  DATA_TYPE *data;
  DATA_TYPE *symmat;
  DATA_TYPE *mean;
  DATA_TYPE *symmat_outputFromGpu;

  data = (DATA_TYPE *)malloc((M + 1) * (N + 1) * sizeof(DATA_TYPE));
  symmat = (DATA_TYPE *)malloc((M + 1) * (M + 1) * sizeof(DATA_TYPE));
  mean = (DATA_TYPE *)malloc((M + 1) * sizeof(DATA_TYPE));
  symmat_outputFromGpu =
      (DATA_TYPE *)malloc((M + 1) * (M + 1) * sizeof(DATA_TYPE));

  fprintf(stdout, "<< Covariance Computation >>\n");

  init_arrays(data);

  t_start = rtclock();
  GPU__covariance(data, symmat_outputFromGpu, mean);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  init_arrays(data);

  t_start = rtclock();
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(symmat, symmat_outputFromGpu);

  free(data);
  free(symmat);
  free(mean);
  free(symmat_outputFromGpu);

  return 0;
}

