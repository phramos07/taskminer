/**
 * 3mm.c: This file was adapted from PolyBench/GPU 1.0 test suite
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

/* Problem size. */
#ifdef MINI_DATASET
#define NI 32
#define NJ 32
#define NK 32
#define NL 32
#define NM 32
#endif

#ifdef SMALL_DATASET
#define NI 128
#define NJ 128
#define NK 128
#define NL 128
#define NM 128
#endif

#ifdef MEDIUM_DATASET
#define NI 512
#define NJ 512
#define NK 512
#define NL 512
#define NM 512
#endif

#ifdef LARGE_DATASET
#define NI 1024
#define NJ 1024
#define NK 1024
#define NL 1024
#define NM 1024
#endif

#ifdef EXTRALARGE_DATASET
#define NI 2048
#define NJ 2048
#define NK 2048
#define NL 2048
#define NM 2048
#endif

#define GPU_DEVICE 1

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE *restrict A, DATA_TYPE *restrict B,
                DATA_TYPE *restrict C, DATA_TYPE *restrict D) {
  int i, j;

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NK; j++) {
      A[i * NK + j] = ((DATA_TYPE)i * j) / NI;
    }
  }

  for (i = 0; i < NK; i++) {
    for (j = 0; j < NJ; j++) {
      B[i * NJ + j] = ((DATA_TYPE)i * (j + 1)) / NJ;
    }
  }

  for (i = 0; i < NJ; i++) {
    for (j = 0; j < NM; j++) {
      C[i * NM + j] = ((DATA_TYPE)i * (j + 3)) / NL;
    }
  }

  for (i = 0; i < NM; i++) {
    for (j = 0; j < NL; j++) {
      D[i * NL + j] = ((DATA_TYPE)i * (j + 2)) / NK;
    }
  }
}

void compareResults(DATA_TYPE *restrict G,
                    DATA_TYPE *restrict G_outputFromGpu) {
  int i, j, fail;
  fail = 0;

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NL; j++) {
      if (percentDiff(G[i * NL + j], G_outputFromGpu[i * NL + j]) >
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

void mm3_cpu(DATA_TYPE *restrict A, DATA_TYPE *restrict B,
             DATA_TYPE *restrict C, DATA_TYPE *restrict D,
             DATA_TYPE *restrict E, DATA_TYPE *restrict F,
             DATA_TYPE *restrict G) {
  int i, j, k;

  /* E := A*B */
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      E[i * NJ + j] = 0;
      for (k = 0; k < NK; ++k) {
        E[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
      }
    }
  }

  /* F := C*D */
  for (i = 0; i < NJ; i++) {
    for (j = 0; j < NL; j++) {
      F[i * NL + j] = 0;
      for (k = 0; k < NM; ++k) {
        F[i * NL + j] += C[i * NM + k] * D[k * NL + j];
      }
    }
  }

  /* G := E*F */
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NL; j++) {
      G[i * NL + j] = 0;
      for (k = 0; k < NJ; ++k) {
        G[i * NL + j] += E[i * NJ + k] * F[k * NL + j];
      }
    }
  }
}

void GPU__mm3(DATA_TYPE *restrict A, DATA_TYPE *restrict B,
              DATA_TYPE *restrict C, DATA_TYPE *restrict D,
              DATA_TYPE *restrict E, DATA_TYPE *restrict F,
              DATA_TYPE *restrict G) {
  int i, j, k;

/* E := A*B */
#pragma omp parallel for
  char RST_AI1 = 0;
  RST_AI1 |= !((A + 0 > B + 16384)
  || (B + 0 > A + 16384));
  RST_AI1 |= !((A + 0 > C + 16384)
  || (C + 0 > A + 16384));
  RST_AI1 |= !((A + 0 > D + 16384)
  || (D + 0 > A + 16384));
  RST_AI1 |= !((A + 0 > E + 16384)
  || (E + 0 > A + 16384));
  RST_AI1 |= !((A + 0 > F + 16384)
  || (F + 0 > A + 16384));
  RST_AI1 |= !((A + 0 > G + 16384)
  || (G + 0 > A + 16384));
  RST_AI1 |= !((B + 0 > C + 16384)
  || (C + 0 > B + 16384));
  RST_AI1 |= !((B + 0 > D + 16384)
  || (D + 0 > B + 16384));
  RST_AI1 |= !((B + 0 > E + 16384)
  || (E + 0 > B + 16384));
  RST_AI1 |= !((B + 0 > F + 16384)
  || (F + 0 > B + 16384));
  RST_AI1 |= !((B + 0 > G + 16384)
  || (G + 0 > B + 16384));
  RST_AI1 |= !((C + 0 > D + 16384)
  || (D + 0 > C + 16384));
  RST_AI1 |= !((C + 0 > E + 16384)
  || (E + 0 > C + 16384));
  RST_AI1 |= !((C + 0 > F + 16384)
  || (F + 0 > C + 16384));
  RST_AI1 |= !((C + 0 > G + 16384)
  || (G + 0 > C + 16384));
  RST_AI1 |= !((D + 0 > E + 16384)
  || (E + 0 > D + 16384));
  RST_AI1 |= !((D + 0 > F + 16384)
  || (F + 0 > D + 16384));
  RST_AI1 |= !((D + 0 > G + 16384)
  || (G + 0 > D + 16384));
  RST_AI1 |= !((E + 0 > F + 16384)
  || (F + 0 > E + 16384));
  RST_AI1 |= !((E + 0 > G + 16384)
  || (G + 0 > E + 16384));
  RST_AI1 |= !((F + 0 > G + 16384)
  || (G + 0 > F + 16384));
  #pragma omp target data map(to: A[0:16384],B[0:16384],C[0:16384],D[0:16384]) map(tofrom: E[0:16384],F[0:16384],G[0:16384]) if(!RST_AI1)
  {
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      E[i * NJ + j] = 0;
      for (k = 0; k < NK; ++k) {
        E[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
      }
    }
  }

/* F := C*D */
#pragma omp parallel for
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < NJ; i++) {
    for (j = 0; j < NL; j++) {
      F[i * NL + j] = 0;
      for (k = 0; k < NM; ++k) {
        F[i * NL + j] += C[i * NM + k] * D[k * NL + j];
      }
    }
  }

/* G := E*F */
#pragma omp parallel for
  #pragma omp target if(!RST_AI1)
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NL; j++) {
      G[i * NL + j] = 0;
      for (k = 0; k < NJ; ++k) {
        G[i * NL + j] += E[i * NJ + k] * F[k * NL + j];
      }
    }
  }
}
}

int main(int argc, char **argv) {
  double t_start, t_end;

  DATA_TYPE *A;
  DATA_TYPE *B;
  DATA_TYPE *C;
  DATA_TYPE *D;
  DATA_TYPE *E;
  DATA_TYPE *F;
  DATA_TYPE *G;
  DATA_TYPE *G_outputFromGpu;

  A = (DATA_TYPE *)malloc(NI * NK * sizeof(DATA_TYPE));
  B = (DATA_TYPE *)malloc(NK * NJ * sizeof(DATA_TYPE));
  C = (DATA_TYPE *)malloc(NJ * NM * sizeof(DATA_TYPE));
  D = (DATA_TYPE *)malloc(NM * NL * sizeof(DATA_TYPE));
  E = (DATA_TYPE *)malloc(NI * NJ * sizeof(DATA_TYPE));
  F = (DATA_TYPE *)malloc(NJ * NL * sizeof(DATA_TYPE));
  G = (DATA_TYPE *)malloc(NI * NL * sizeof(DATA_TYPE));
  G_outputFromGpu = (DATA_TYPE *)malloc(NI * NL * sizeof(DATA_TYPE));

  fprintf(
      stdout,
      "<< Linear Algebra: 3 Matrix Multiplications (E=A.B; F=C.D; G=E.F) >>\n");

  init_array(A, B, C, D);

  t_start = rtclock();
  GPU__mm3(A, B, C, D, E, F, G_outputFromGpu);
  t_end = rtclock();

  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  mm3_cpu(A, B, C, D, E, F, G);
  t_end = rtclock();

  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(G, G_outputFromGpu);

  free(A);
  free(B);
  free(C);
  free(D);
  free(E);
  free(F);
  free(G);
  free(G_outputFromGpu);

  return 0;
}

