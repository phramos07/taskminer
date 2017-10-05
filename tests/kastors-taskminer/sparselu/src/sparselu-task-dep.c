#include <omp.h>
/**********************************************************************************************/
/*  This program is part of the Barcelona OpenMP Tasks Suite */
/*  Copyright (C) 2009 Barcelona Supercomputing Center - Centro Nacional de
 * Supercomputacion  */
/*  Copyright (C) 2009 Universitat Politecnica de Catalunya */
/*                                                                                            */
/*  This program is free software; you can redistribute it and/or modify */
/*  it under the terms of the GNU General Public License as published by */
/*  the Free Software Foundation; either version 2 of the License, or */
/*  (at your option) any later version. */
/*                                                                                            */
/*  This program is distributed in the hope that it will be useful, */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the */
/*  GNU General Public License for more details. */
/*                                                                                            */
/*  You should have received a copy of the GNU General Public License */
/*  along with this program; if not, write to the Free Software */
/*  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
 * USA            */
/**********************************************************************************************/

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>
#include "../include/sparselu.h"

void sparselu_par_call(float **BENCH, int matrix_size, int submatrix_size) {
  int ii, jj, kk;
  {
    #pragma omp parallel
    #pragma omp single
    for (kk = 0; kk < matrix_size; kk++) {
      long long int TM5[2];
      TM5[0] = kk * matrix_size;
      TM5[1] = TM5[0] + kk;
      #pragma omp task depend(inout:BENCH[TM5[1]])
      lu0(BENCH[kk * matrix_size + kk], submatrix_size);
      for (jj = kk + 1; jj < matrix_size; jj++) {
        if (BENCH[kk * matrix_size + jj] != NULL) {
          long long int TM7[3];
          TM7[0] = kk * matrix_size;
          TM7[1] = TM7[0] + kk;
          TM7[2] = TM7[0] + jj;
          #pragma omp task depend(in:BENCH[TM7[1]]) depend(inout:BENCH[TM7[2]])
          fwd(BENCH[kk * matrix_size + kk], BENCH[kk * matrix_size + jj],
              submatrix_size);
        }
      }
      for (ii = kk + 1; ii < matrix_size; ii++) {
        if (BENCH[ii * matrix_size + kk] != NULL) {
          long long int TM10[4];
          TM10[0] = kk * matrix_size;
          TM10[1] = TM10[0] + kk;
          TM10[2] = ii * matrix_size;
          TM10[3] = TM10[2] + kk;
          #pragma omp task depend(in:BENCH[TM10[1]]) depend(inout:BENCH[TM10[3]])
          bdiv(BENCH[kk * matrix_size + kk], BENCH[ii * matrix_size + kk],
               submatrix_size);
        }
      }
      for (ii = kk + 1; ii < matrix_size; ii++) {
        if (BENCH[ii * matrix_size + kk] != NULL) {
          for (jj = kk + 1; jj < matrix_size; jj++) {
            if (BENCH[kk * matrix_size + jj] != NULL) {
              if (BENCH[ii * matrix_size + jj] == NULL) {
                BENCH[ii * matrix_size + jj] =
                    #pragma omp task
                    allocate_clean_block(submatrix_size);
              }
              long long int TM15[5];
              TM15[0] = ii * matrix_size;
              TM15[1] = TM15[0] + kk;
              TM15[2] = kk * matrix_size;
              TM15[3] = TM15[2] + jj;
              TM15[4] = TM15[0] + jj;
              #pragma omp task depend(in:BENCH[TM15[1]],BENCH[TM15[3]]) depend(inout:BENCH[TM15[4]])
              bmod(BENCH[ii * matrix_size + kk], BENCH[kk * matrix_size + jj],
                   BENCH[ii * matrix_size + jj], submatrix_size);
            }
          }
        }
      }
    }
  }
}

