#include <omp.h>
#ifndef taskminerutils
#define taskminerutils
static int taskminer_depth_cutoff = 0;
#define DEPTH_CUTOFF omp_get_num_threads()
char cutoff_test = 0;
#endif

#include <stdio.h>
#include <stdlib.h>
#include "backprop.h"

extern layer_size;

load(net) BPNN *net;
{
  float *units;
  int nr, nc, imgsize, i, j, k;

  nr = layer_size;

  imgsize = nr * nc;
  units = net->input_units;

  k = 1;
  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < nr; i++) {
    {
    long long int TM11[17];
    TM11[0] = units[4] / 96;
    TM11[1] = (TM11[0] > 0);
    TM11[2] = (TM11[1] ? TM11[0] : 0);
    TM11[3] = nr > 0;
    TM11[4] = (TM11[3] ? nr : 0);
    TM11[5] = 4 * TM11[4];
    TM11[6] = units[4] + TM11[5];
    TM11[7] = TM11[6] + 96;
    TM11[8] = TM11[7] / 96;
    TM11[9] = (TM11[8] > 0);
    TM11[10] = (TM11[9] ? TM11[8] : 0);
    TM11[11] = TM11[10] - TM11[2];
    TM11[12] = (TM11[11] > 0);
    TM11[13] = TM11[2] + TM11[11];
    TM11[14] = -1 * TM11[11];
    TM11[15] = TM11[12] ? TM11[2] : TM11[13];
    TM11[16] = TM11[12] ? TM11[11] : TM11[14];
    int tm_cost1 = (22);
    #pragma omp task depend(inout: net[TM11[15]:TM11[16]]) if(tm_cost1 > 41)
    {
    units[k] = (float)rand() / RAND_MAX;
    k++;
  }
  }
  }
}

