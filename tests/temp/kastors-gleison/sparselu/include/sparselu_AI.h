#include <omp.h>
#ifndef SPARSELU_H
#define SPARSELU_H

#define EPSILON 1.0E-6



float * allocate_clean_block(int submatrix_size);
/***********************************************************************
 * lu0:
 **********************************************************************/
void lu0(float *diag, int submatrix_size)
{
    int i, j, k;

    for (k=0; k<submatrix_size; k++)
        for (i=k+1; i<submatrix_size; i++)
        {
            diag[i*submatrix_size+k] = diag[i*submatrix_size+k] / diag[k*submatrix_size+k];
            for (j=k+1; j<submatrix_size; j++)
                diag[i*submatrix_size+j] = diag[i*submatrix_size+j] - diag[i*submatrix_size+k] * diag[k*submatrix_size+j];
        }
}

/***********************************************************************
 * bdiv:
 **********************************************************************/
void bdiv(float *diag, float *row, int submatrix_size)
{
    int i, j, k;
    for (i=0; i<submatrix_size; i++)
        for (k=0; k<submatrix_size; k++)
        {
            row[i*submatrix_size+k] = row[i*submatrix_size+k] / diag[k*submatrix_size+k];
            for (j=k+1; j<submatrix_size; j++)
                row[i*submatrix_size+j] = row[i*submatrix_size+j] - row[i*submatrix_size+k]*diag[k*submatrix_size+j];
        }
}
/***********************************************************************
 * bmod:
 **********************************************************************/
void bmod(float *row, float *col, float *inner, int submatrix_size)
{
    int i, j, k;
    for (i=0; i<submatrix_size; i++)
        for (j=0; j<submatrix_size; j++)
            for (k=0; k<submatrix_size; k++)
                inner[i*submatrix_size+j] = inner[i*submatrix_size+j] - row[i*submatrix_size+k]*col[k*submatrix_size+j];
}
/***********************************************************************
 * fwd:
 **********************************************************************/
void fwd(float *diag, float *col, int submatrix_size)
{
    int i, j, k;
    for (j=0; j<submatrix_size; j++)
        for (k=0; k<submatrix_size; k++)
            for (i=k+1; i<submatrix_size; i++)
                col[i*submatrix_size+j] = col[i*submatrix_size+j] - diag[i*submatrix_size+k]*col[k*submatrix_size+j];
}


void sparselu_seq_call(float **BENCH, int matrix_size, int submatrix_size);
void sparselu_par_call(float **BENCH, int matrix_size, int submatrix_size);

#endif

