/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* 
 * Base C implementation of MM
 */

#include <iostream>

void basicSgemmGPU( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc )
{
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    return;
  }
  
  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    return;
  }

	int mm, nn, i;
	#pragma omp target device(1)
	#pragma omp target map(to: A[:m*k], B[:n*k]) map(tofrom: C[:m*n])
  #pragma omp parallel for
  long long int AI1[18];
  AI1[0] = m > 0;
  AI1[1] = (AI1[0] ? m : 0);
  AI1[2] = k > 0;
  AI1[3] = (AI1[2] ? k : 0);
  AI1[4] = lda * AI1[3];
  AI1[5] = AI1[1] + AI1[4];
  AI1[6] = AI1[5] * 4;
  AI1[7] = AI1[6] / 4;
  AI1[8] = n > 0;
  AI1[9] = (AI1[8] ? n : 0);
  AI1[10] = ldb * AI1[3];
  AI1[11] = AI1[9] + AI1[10];
  AI1[12] = AI1[11] * 4;
  AI1[13] = AI1[12] / 4;
  AI1[14] = ldc * AI1[9];
  AI1[15] = AI1[1] + AI1[14];
  AI1[16] = AI1[15] * 4;
  AI1[17] = AI1[16] / 4;
  #pragma acc data copy(A[0:AI1[7]],B[0:AI1[13]],C[0:AI1[17]])
  #pragma acc kernels
  #pragma acc loop independent
  for (mm = 0; mm < m; ++mm) {
    for (nn = 0; nn < n; ++nn) {
      float c = 0.0f;
      for (i = 0; i < k; ++i) {
        float a = A[mm + i * lda]; 
        float b = B[nn + i * ldb];
        c += a * b;
      }
      C[mm+nn*ldc] = C[mm+nn*ldc] * beta + alpha * c;
    }
  }
}

void basicSgemmCPU( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc )
{
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    return;
  }
  
  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    return;
  }

	int mm, nn, i;
  long long int AI1[18];
  AI1[0] = m > 0;
  AI1[1] = (AI1[0] ? m : 0);
  AI1[2] = k > 0;
  AI1[3] = (AI1[2] ? k : 0);
  AI1[4] = lda * AI1[3];
  AI1[5] = AI1[1] + AI1[4];
  AI1[6] = AI1[5] * 4;
  AI1[7] = AI1[6] / 4;
  AI1[8] = n > 0;
  AI1[9] = (AI1[8] ? n : 0);
  AI1[10] = ldb * AI1[3];
  AI1[11] = AI1[9] + AI1[10];
  AI1[12] = AI1[11] * 4;
  AI1[13] = AI1[12] / 4;
  AI1[14] = ldc * AI1[9];
  AI1[15] = AI1[1] + AI1[14];
  AI1[16] = AI1[15] * 4;
  AI1[17] = AI1[16] / 4;
  #pragma acc data copy(A[0:AI1[7]],B[0:AI1[13]],C[0:AI1[17]])
  #pragma acc kernels
  #pragma acc loop independent
  for (mm = 0; mm < m; ++mm) {
    for (nn = 0; nn < n; ++nn) {
      float c = 0.0f;
      for (i = 0; i < k; ++i) {
        float a = A[mm + i * lda]; 
        float b = B[nn + i * ldb];
        c += a * b;
      }
      C[mm+nn*ldc] = C[mm+nn*ldc] * beta + alpha * c;
    }
  }
}

