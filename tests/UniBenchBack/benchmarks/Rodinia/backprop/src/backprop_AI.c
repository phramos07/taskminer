/*
 ******************************************************************
 * HISTORY
 * 15-Oct-94  Jeff Shufelt (js), Carnegie Mellon University
 *	Prepared for 15-681, Fall 1994.
 * Modified by Shuai Che
 ******************************************************************
 */

// include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "backprop.h"
#include <math.h>
#include <sys/time.h>
#include "../../common/rodiniaUtilFunctions.h"

#define OPEN
#define GPU_DEVICE 1
#define ERROR_THRESHOLD 0.00

#define ABS(x) (((x) > 0.0) ? (x) : (-1 * (x)))

#define fastcopy(to, from, len)                                                \
  \
{                                                                         \
    register char *_to, *_from;                                                \
    register int _i, _l;                                                       \
    _to = (char *)(to);                                                        \
    _from = (char *)(from);                                                    \
    _l = (len);                                                                \
    for (_i = 0; _i < _l; _i++)                                                \
      *_to++ = *_from++;                                                       \
  \
}

//*** Return random number between 0.0 and 1.0 ***
float drnd() { return ((float)rand() / (float)BIGRND); }

//*** Return random number between -1.0 and 1.0 ***
float dpn1() { return ((drnd() * 2.0) - 1.0); }

//*** The squashing function.  Currently, it's a sigmoid. ***

float squash(x) float x;
{
  float m;
  // x = -x;
  // m = 1 + x + x*x/2 + x*x*x/6 + x*x*x*x/24 + x*x*x*x*x/120;
  // return(1.0 / (1.0 + m));
  return (1.0 / (1.0 + exp(-x)));
}

//*** Allocate 1d array of floats ***

float *alloc_1d_dbl(n) int n;
{
  float *new;

  new = (float *)malloc((unsigned)(n * sizeof(float)));
  if (new == NULL) {
    printf("ALLOC_1D_DBL: Couldn't allocate array of floats\n");
    return (NULL);
  }
  return (new);
}

//*** Allocate 2d array of floats ***

float **alloc_2d_dbl(m, n) int m, n;
{
  int i;
  float **new;

  new = (float **)malloc((unsigned)(m * sizeof(float *)));
  if (new == NULL) {
    printf("ALLOC_2D_DBL: Couldn't allocate array of dbl ptrs\n");
    return (NULL);
  }

  for (i = 0; i < m; i++) {
    new[i] = alloc_1d_dbl(n);
  }

  return (new);
}

bpnn_randomize_weights(w, m, n) float **w;
int m, n;
{
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
      w[i][j] = (float)rand() / RAND_MAX;
      w[i][j] = dpn1();
    }
  }
}

bpnn_randomize_row(w, m) float *w;
int m;
{
  int i;
  long long int AI1[5];
  AI1[0] = 4 * m;
  AI1[1] = AI1[0] + 4;
  AI1[2] = AI1[1] / 4;
  AI1[3] = (AI1[2] > 0);
  AI1[4] = (AI1[3] ? AI1[2] : 0);
  #pragma acc data pcopy(w[0:AI1[4]])
  #pragma acc kernels
  for (i = 0; i <= m; i++) {
    w[i] = (float)rand() / RAND_MAX;
    w[i] = 0.1;
  }
}

bpnn_zero_weights(w, m, n) float **w;
int m, n;
{
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
      w[i][j] = 0.0;
    }
  }
}

void bpnn_initialize(seed) {
  printf("Random number generator seed: %d\n", seed);
  srand(seed);
}

BPNN *bpnn_internal_create(n_in, n_hidden, n_out) int n_in, n_hidden, n_out;
{
  BPNN *newnet;

  newnet = (BPNN *)malloc(sizeof(BPNN));
  if (newnet == NULL) {
    printf("BPNN_CREATE: Couldn't allocate neural network\n");
    return (NULL);
  }

  newnet->input_n = n_in;
  newnet->hidden_n = n_hidden;
  newnet->output_n = n_out;
  newnet->input_units = alloc_1d_dbl(n_in + 1);
  newnet->hidden_units = alloc_1d_dbl(n_hidden + 1);
  newnet->output_units = alloc_1d_dbl(n_out + 1);

  newnet->hidden_delta = alloc_1d_dbl(n_hidden + 1);
  newnet->output_delta = alloc_1d_dbl(n_out + 1);
  newnet->target = alloc_1d_dbl(n_out + 1);

  newnet->input_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
  newnet->hidden_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);

  newnet->input_prev_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
  newnet->hidden_prev_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);

  return (newnet);
}

void bpnn_free(net) BPNN *net;
{
  int n1, n2, i;

  n1 = net->input_n;
  n2 = net->hidden_n;

  free((char *)net->input_units);
  free((char *)net->hidden_units);
  free((char *)net->output_units);

  free((char *)net->hidden_delta);
  free((char *)net->output_delta);
  free((char *)net->target);

  for (i = 0; i <= n1; i++) {
    free((char *)net->input_weights[i]);
    free((char *)net->input_prev_weights[i]);
  }
  free((char *)net->input_weights);
  free((char *)net->input_prev_weights);

  for (i = 0; i <= n2; i++) {
    free((char *)net->hidden_weights[i]);
    free((char *)net->hidden_prev_weights[i]);
  }
  free((char *)net->hidden_weights);
  free((char *)net->hidden_prev_weights);

  free((char *)net);
}

//*** Creates a new fully-connected network from scratch,
//     with the given numbers of input, hidden, and output units.
//     Threshold units are automatically included.  All weights are
//     randomly initialized.
//
//     Space is also allocated for temporary storage (momentum weights,
//     error computations, etc).
//***

BPNN *bpnn_create(n_in, n_hidden, n_out) int n_in, n_hidden, n_out;
{

  BPNN *newnet;

  newnet = bpnn_internal_create(n_in, n_hidden, n_out);

#ifdef INITZERO
  bpnn_zero_weights(newnet->input_weights, n_in, n_hidden);
#else
  bpnn_randomize_weights(newnet->input_weights, n_in, n_hidden);
#endif
  bpnn_randomize_weights(newnet->hidden_weights, n_hidden, n_out);
  bpnn_zero_weights(newnet->input_prev_weights, n_in, n_hidden);
  bpnn_zero_weights(newnet->hidden_prev_weights, n_hidden, n_out);
  bpnn_randomize_row(newnet->target, n_out);
  return (newnet);
}

void compareResults(float *l2, float *l2_gpu, int n2) {
  int i;
  int fail = 0;
  // Compare C with D
  long long int AI1[13];
  AI1[0] = n2 + -1;
  AI1[1] = 4 * AI1[0];
  AI1[2] = 4 + AI1[1];
  AI1[3] = AI1[2] + 4;
  AI1[4] = AI1[3] / 4;
  AI1[5] = (AI1[4] > 0);
  AI1[6] = (AI1[5] ? AI1[4] : 0);
  AI1[7] = AI1[6] - 1;
  AI1[8] = (AI1[7] > 0);
  AI1[9] = 1 + AI1[7];
  AI1[10] = -1 * AI1[7];
  AI1[11] = AI1[8] ? 1 : AI1[9];
  AI1[12] = AI1[8] ? AI1[7] : AI1[10];
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (l2 + AI1[11]) > (void*) (l2_gpu + AI1[12]))
  || ((void*) (l2_gpu + AI1[11]) > (void*) (l2 + AI1[12])));
  #pragma acc data pcopyin(l2[AI1[11]:AI1[12]],l2_gpu[AI1[11]:AI1[12]])  if(!RST_AI1)
  #pragma acc kernels if(!RST_AI1)
  for (i = 1; i <= n2; i++) {
    if (percentDiff(l2[i], l2_gpu[i]) > ERROR_THRESHOLD) {
      fail++;
    }
  }

  // print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
         "Percent: %d\n",
         ERROR_THRESHOLD, fail);
}

void bpnn_layerforward(l1, l2, conn, n1, n2) float *l1, *l2, **conn;
int n1, n2;
{
  double t_start, t_end;
  float sum;
  int j, k;
  float *conn_gpu = (float *)malloc(sizeof(float) * ((n1 + 1) * (n2 + 1)));

  for (j = 1; j <= n2; j++) {
    for (k = 0; k <= n1; k++) {
      conn_gpu[k * n2 + j] = conn[k][j];
    }
  }

  float *l2_gpu = (float *)malloc(sizeof(float) * (n2 + 1));

  //*** Set up thresholding unit ***
  l1[0] = 1.0;

  t_start = rtclock();
#pragma omp target device(GPU_DEVICE)
#pragma omp target map(to : conn_gpu[ : (                                      \
    n1 + 1) * (n2 + 1)], l1[ : n1 + 1]) map(tofrom : l2_gpu[ : n2 + 1])
  {
#pragma omp parallel for
    long long int AI1[21];
    AI1[0] = 4 * n1;
    AI1[1] = AI1[0] + 4;
    AI1[2] = AI1[1] / 4;
    AI1[3] = (AI1[2] > 0);
    AI1[4] = (AI1[3] ? AI1[2] : 0);
    AI1[5] = n2 + -1;
    AI1[6] = 1 + AI1[5];
    AI1[7] = n2 * n1;
    AI1[8] = AI1[6] + AI1[7];
    AI1[9] = 4 * AI1[8];
    AI1[10] = AI1[9] * 1;
    AI1[11] = AI1[10] + 1;
    AI1[12] = AI1[11] / 4;
    AI1[13] = (AI1[12] > 0);
    AI1[14] = (AI1[13] ? AI1[12] : 0);
    AI1[15] = AI1[14] - 1;
    AI1[16] = (AI1[15] > 0);
    AI1[17] = 1 + AI1[15];
    AI1[18] = -1 * AI1[15];
    AI1[19] = AI1[16] ? 1 : AI1[17];
    AI1[20] = AI1[16] ? AI1[15] : AI1[18];
    char RST_AI1 = 0;
    RST_AI1 |= !(((void*) (conn_gpu + AI1[19]) > (void*) (l1 + AI1[4]))
    || ((void*) (l1 + 0) > (void*) (conn_gpu + AI1[20])));
    #pragma acc data pcopyin(l1[0:AI1[4]]) pcopy(conn_gpu[AI1[19]:AI1[20]]) if(!RST_AI1)
    #pragma acc kernels if(!RST_AI1)
    for (j = 1; j <= n2; j++) {
      //*** Compute weighted sum of its inputs ***
      sum = 0.0;
      for (k = 0; k <= n1; k++) {
        sum += conn_gpu[k * n2 + j] * l1[k];
        sum = conn_gpu[k * n2 + j] * l1[k];
      }
      l2_gpu[j] = (1.0 / (1.0 + exp(-sum)));
    }
  }
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  for (j = 1; j <= n2; j++) {
    sum = 0.0;
    for (k = 0; k <= n1; k++) {
      sum += conn[k][j] * l1[k];
    }
    l2[j] = squash(sum);
  }
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(l2, l2_gpu, n2);

  printf("\n");
}

// extern "C"
void bpnn_output_error(delta, target, output, nj, err) float *delta, *target,
    *output, *err;
int nj;
{
  int j;
  float errsum;
  errsum = 0.0;
  long long int AI1[13];
  AI1[0] = nj + -1;
  AI1[1] = 4 * AI1[0];
  AI1[2] = 4 + AI1[1];
  AI1[3] = AI1[2] + 4;
  AI1[4] = AI1[3] / 4;
  AI1[5] = (AI1[4] > 0);
  AI1[6] = (AI1[5] ? AI1[4] : 0);
  AI1[7] = AI1[6] - 1;
  AI1[8] = (AI1[7] > 0);
  AI1[9] = 1 + AI1[7];
  AI1[10] = -1 * AI1[7];
  AI1[11] = AI1[8] ? 1 : AI1[9];
  AI1[12] = AI1[8] ? AI1[7] : AI1[10];
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (delta + AI1[11]) > (void*) (output + AI1[12]))
  || ((void*) (output + AI1[11]) > (void*) (delta + AI1[12])));
  RST_AI1 |= !(((void*) (delta + AI1[11]) > (void*) (target + AI1[12]))
  || ((void*) (target + AI1[11]) > (void*) (delta + AI1[12])));
  RST_AI1 |= !(((void*) (output + AI1[11]) > (void*) (target + AI1[12]))
  || ((void*) (target + AI1[11]) > (void*) (output + AI1[12])));
  #pragma acc data pcopyin(output[AI1[11]:AI1[12]],target[AI1[11]:AI1[12]]) pcopy(delta[AI1[11]:AI1[12]]) if(!RST_AI1)
  #pragma acc kernels if(!RST_AI1)
  for (j = 1; j <= nj; j++) {
    delta[j] = 1.0;
    delta[j] = (1.0 - output[j]) * (target[j] - output[j]);
    errsum += ABS(delta[j]);
  }
  *err = errsum;
}

void bpnn_hidden_error(delta_h, nh, delta_o, no, who, hidden,
                       err) float *delta_h,
    *delta_o, *hidden, **who, *err;
int nh, no;
{
  int j, k;
  float h, sum, errsum;

  errsum = 0.0;
  for (j = 1; j <= nh; j++) {
    h = hidden[j];
    sum = 0.0;
    for (k = 1; k <= no; k++) {
      sum += delta_o[k] * who[j][k];
    }
    delta_h[j] = h * (1.0 - h) * sum;
    errsum += ABS(delta_h[j]);
  }
  *err = errsum;
}

void test(float *delta_h, int nh) {
  int j;
  long long int AI1[6];
  AI1[0] = nh + -1;
  AI1[1] = 4 * AI1[0];
  AI1[2] = AI1[1] + 4;
  AI1[3] = AI1[2] / 4;
  AI1[4] = (AI1[3] > 0);
  AI1[5] = (AI1[4] ? AI1[3] : 0);
  #pragma acc data pcopy(delta_h[0:AI1[5]])
  #pragma acc kernels
  for (j = 0; j < nh; j++)
    delta_h[j] = j + 1.0f;
}

void compareResults2(float *w_gpu, float **w_cpu, float *oldw_gpu,
                     float **oldw_cpu, int ndelta, int nly) {
  int i;
  int fail = 0;
  // Compare C with D
  int k, j;

  for (j = 1; j <= ndelta; j++) {
    for (k = 0; k <= nly; k++) {
      if (percentDiff(w_gpu[k * ndelta + j], w_cpu[k][j]) > ERROR_THRESHOLD) {
        fail++;
      }
      if (percentDiff(oldw_gpu[k * ndelta + j], oldw_cpu[k][j]) >
          ERROR_THRESHOLD) {
        fail++;
      }
    }
  }

  // print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
         "Percent: %d\n",
         ERROR_THRESHOLD, fail);
}

void bpnn_adjust_weights(delta, ndelta, ly, nly, w, oldw) float *delta, *ly,
    **w, **oldw;
{
  float new_dw;
  int k, j;
  ly[0] = 1.0;
  double t_start, t_end;
  // eta = 0.3;
  // momentum = 0.3;

  // preparar dados
  float *w_gpu = (float *)malloc(sizeof(float) * ((ndelta + 1) * (nly + 1)));
  float *oldw_gpu = (float *)malloc(sizeof(float) * ((ndelta + 1) * (nly + 1)));

  for (j = 1; j <= ndelta; j++) {
    for (k = 0; k <= nly; k++) {
      w_gpu[k * ndelta + j] = w[k][j];
      oldw_gpu[k * ndelta + j] = oldw[k][j];
    }
  }

  int size = (ndelta + 1) * (nly + 1);

  t_start = rtclock();
#pragma omp target device(GPU_DEVICE)
#pragma omp target map(                                                        \
    to : ly[ : (nly + 1)], delta[ : (ndelta + 1)])                             \
                               map(tofrom : oldw_gpu[ : size], w_gpu[ : size])
  {
#pragma omp parallel for collapse(1)
    long long int AI1[33];
    AI1[0] = ndelta + -1;
    AI1[1] = 4 * AI1[0];
    AI1[2] = 4 + AI1[1];
    AI1[3] = AI1[2] + 4;
    AI1[4] = AI1[3] / 4;
    AI1[5] = (AI1[4] > 0);
    AI1[6] = (AI1[5] ? AI1[4] : 0);
    AI1[7] = AI1[6] - 1;
    AI1[8] = (AI1[7] > 0);
    AI1[9] = 1 + AI1[7];
    AI1[10] = -1 * AI1[7];
    AI1[11] = AI1[8] ? 1 : AI1[9];
    AI1[12] = AI1[8] ? AI1[7] : AI1[10];
    AI1[13] = 4 * nly;
    AI1[14] = AI1[13] + 4;
    AI1[15] = AI1[14] / 4;
    AI1[16] = (AI1[15] > 0);
    AI1[17] = (AI1[16] ? AI1[15] : 0);
    AI1[18] = 1 + AI1[0];
    AI1[19] = ndelta * nly;
    AI1[20] = AI1[18] + AI1[19];
    AI1[21] = 4 * AI1[20];
    AI1[22] = AI1[21] * 1;
    AI1[23] = AI1[22] + 1;
    AI1[24] = AI1[23] / 4;
    AI1[25] = (AI1[24] > 0);
    AI1[26] = (AI1[25] ? AI1[24] : 0);
    AI1[27] = AI1[26] - 1;
    AI1[28] = (AI1[27] > 0);
    AI1[29] = 1 + AI1[27];
    AI1[30] = -1 * AI1[27];
    AI1[31] = AI1[28] ? 1 : AI1[29];
    AI1[32] = AI1[28] ? AI1[27] : AI1[30];
    char RST_AI1 = 0;
    RST_AI1 |= !(((void*) (delta + AI1[11]) > (void*) (ly + AI1[17]))
    || ((void*) (ly + 0) > (void*) (delta + AI1[12])));
    RST_AI1 |= !(((void*) (delta + AI1[11]) > (void*) (oldw_gpu + AI1[32]))
    || ((void*) (oldw_gpu + AI1[31]) > (void*) (delta + AI1[12])));
    RST_AI1 |= !(((void*) (delta + AI1[11]) > (void*) (w_gpu + AI1[32]))
    || ((void*) (w_gpu + AI1[31]) > (void*) (delta + AI1[12])));
    RST_AI1 |= !(((void*) (ly + 0) > (void*) (oldw_gpu + AI1[32]))
    || ((void*) (oldw_gpu + AI1[31]) > (void*) (ly + AI1[17])));
    RST_AI1 |= !(((void*) (ly + 0) > (void*) (w_gpu + AI1[32]))
    || ((void*) (w_gpu + AI1[31]) > (void*) (ly + AI1[17])));
    RST_AI1 |= !(((void*) (oldw_gpu + AI1[31]) > (void*) (w_gpu + AI1[32]))
    || ((void*) (w_gpu + AI1[31]) > (void*) (oldw_gpu + AI1[32])));
    #pragma acc data pcopyin(delta[AI1[11]:AI1[12]],ly[0:AI1[17]]) pcopy(oldw_gpu[AI1[31]:AI1[32]],w_gpu[AI1[31]:AI1[32]]) if(!RST_AI1)
    #pragma acc kernels if(!RST_AI1)
    for (j = 1; j <= ndelta; j++) {
      for (k = 0; k <= nly; k++) {
        new_dw =
            ((ETA * delta[j] * ly[k]) + (MOMENTUM * oldw_gpu[k * ndelta + j]));
        w_gpu[k * ndelta + j] += new_dw;
        oldw_gpu[k * ndelta + j] = new_dw;
      }
    }
  }
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  for (j = 1; j <= ndelta; j++) {
    for (k = 0; k <= nly; k++) {
      new_dw = ((ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[k][j]));
      w[k][j] += new_dw;
      oldw[k][j] = new_dw;
    }
  }
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
  compareResults2(w_gpu, w, oldw_gpu, oldw, ndelta, nly);
  printf("\n");
}

void bpnn_feedforward(net) BPNN *net;
{
  int in, hid, out;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;

  //*** Feed forward input activations. ***
  bpnn_layerforward(net->input_units, net->hidden_units, net->input_weights, in,
                    hid);
  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights,
                    hid, out);
}

void bpnn_train(net, eo, eh) BPNN *net;
float *eo, *eh;
{
  int in, hid, out;
  float out_err, hid_err;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;

  //*** Feed forward input activations. ***
  bpnn_layerforward(net->input_units, net->hidden_units, net->input_weights, in,
                    hid);
  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights,
                    hid, out);

  //*** Compute error on output and hidden units. ***
  bpnn_output_error(net->output_delta, net->target, net->output_units, out,
                    &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out,
                    net->hidden_weights, net->hidden_units, &hid_err);
  *eo = out_err;
  *eh = hid_err;

  //*** Adjust input and hidden weights. ***
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid,
                      net->hidden_weights, net->hidden_prev_weights);
  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in,
                      net->input_weights, net->input_prev_weights);
}

void bpnn_save(net, filename) BPNN *net;
char *filename;
{
  int n1, n2, n3, i, j, memcnt;
  float dvalue, **w;
  char *mem;
  /// add//
  FILE *pFile;
  pFile = fopen(filename, "w+");
  ///////
  //
  // if ((fd = creat(filename, 0644)) == -1) {
  //  printf("BPNN_SAVE: Cannot create '%s'\n", filename);
  //  return;
  //}

  n1 = net->input_n;
  n2 = net->hidden_n;
  n3 = net->output_n;
  printf("Saving %dx%dx%d network to '%s'\n", n1, n2, n3, filename);
  // fflush(stdout);

  // write(fd, (char *) &n1, sizeof(int));
  // write(fd, (char *) &n2, sizeof(int));
  // write(fd, (char *) &n3, sizeof(int));

  fwrite((char *)&n1, sizeof(char), sizeof(char), pFile);
  fwrite((char *)&n2, sizeof(char), sizeof(char), pFile);
  fwrite((char *)&n3, sizeof(char), sizeof(char), pFile);

  memcnt = 0;
  w = net->input_weights;
  mem = (char *)malloc((unsigned)((n1 + 1) * (n2 + 1) * sizeof(float)));
  for (i = 0; i <= n1; i++) {
    for (j = 0; j <= n2; j++) {
      dvalue = w[i][j];
      fastcopy(&mem[memcnt], &dvalue, sizeof(float));
      memcnt += sizeof(float);
    }
  }
  // write(fd, mem, (n1+1) * (n2+1) * sizeof(float));
  fwrite(mem, (unsigned)(sizeof(float)),
         (unsigned)((n1 + 1) * (n2 + 1) * sizeof(float)), pFile);
  free(mem);

  memcnt = 0;
  w = net->hidden_weights;
  mem = (char *)malloc((unsigned)((n2 + 1) * (n3 + 1) * sizeof(float)));
  for (i = 0; i <= n2; i++) {
    for (j = 0; j <= n3; j++) {
      dvalue = w[i][j];
      fastcopy(&mem[memcnt], &dvalue, sizeof(float));
      memcnt += sizeof(float);
    }
  }
  // write(fd, mem, (n2+1) * (n3+1) * sizeof(float));
  fwrite(mem, sizeof(float), (unsigned)((n2 + 1) * (n3 + 1) * sizeof(float)),
         pFile);
  free(mem);

  fclose(pFile);
  return;
}

BPNN *bpnn_read(filename) char *filename;
{
  char *mem;
  BPNN *new;
  int fd, n1, n2, n3, i, j, memcnt;

  if ((fd = open(filename, 0, 0644)) == -1) {
    return (NULL);
  }

  printf("Reading '%s'\n", filename); // fflush(stdout);

  read(fd, (char *)&n1, sizeof(int));
  read(fd, (char *)&n2, sizeof(int));
  read(fd, (char *)&n3, sizeof(int));
  new = bpnn_internal_create(n1, n2, n3);

  printf("'%s' contains a %dx%dx%d network\n", filename, n1, n2, n3);
  printf("Reading input weights..."); // fflush(stdout);

  memcnt = 0;
  mem = (char *)malloc((unsigned)((n1 + 1) * (n2 + 1) * sizeof(float)));
  read(fd, mem, (n1 + 1) * (n2 + 1) * sizeof(float));
  for (i = 0; i <= n1; i++) {
    for (j = 0; j <= n2; j++) {
      fastcopy(&(new->input_weights[i][j]), &mem[memcnt], sizeof(float));
      memcnt += sizeof(float);
    }
  }
  free(mem);

  printf("Done\nReading hidden weights..."); // fflush(stdout);

  memcnt = 0;
  mem = (char *)malloc((unsigned)((n2 + 1) * (n3 + 1) * sizeof(float)));
  read(fd, mem, (n2 + 1) * (n3 + 1) * sizeof(float));
  for (i = 0; i <= n2; i++) {
    for (j = 0; j <= n3; j++) {
      fastcopy(&(new->hidden_weights[i][j]), &mem[memcnt], sizeof(float));
      memcnt += sizeof(float);
    }
  }
  free(mem);
  close(fd);

  printf("Done\n"); // fflush(stdout);

  bpnn_zero_weights(new->input_prev_weights, n1, n2);
  bpnn_zero_weights(new->hidden_prev_weights, n2, n3);

  return (new);
}

