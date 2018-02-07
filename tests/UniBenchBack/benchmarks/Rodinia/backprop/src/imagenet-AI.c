
#include <stdio.h>
#include <stdlib.h>
#include "backprop.h"

extern layer_size;

load(net)
BPNN *net;
{
  float *units;
  int nr, nc, imgsize, i, j, k;

  nr = layer_size;
  
  imgsize = nr * nc;
  units = net->input_units;

  k = 1;
  long long int AI1[6];
  AI1[0] = units[4] / 96;
  AI1[1] = nr > 0;
  AI1[2] = (AI1[1] ? nr : 0);
  AI1[3] = 4 * AI1[2];
  AI1[4] = units[4] + AI1[3];
  AI1[5] = AI1[4] / 96;
  #pragma acc data copy(net[AI1[0]:AI1[5]])
  #pragma acc kernels
  #pragma acc loop independent
  for (i = 0; i < nr; i++) {
	  units[k] = (float) rand()/RAND_MAX ;
	  k++;
    }
}

