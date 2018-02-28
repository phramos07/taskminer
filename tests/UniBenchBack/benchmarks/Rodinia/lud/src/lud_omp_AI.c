#include <omp.h>
#ifndef taskminerutils
#define taskminerutils
static int taskminer_depth_cutoff = 0;
#define DEPTH_CUTOFF omp_get_num_threads()
char cutoff_test = 0;
#endif
#include <stdio.h>
// include <omp.h>
#define GPU_DEVICE 1

void lud_omp_cpu(float *a, int size) {
  int i, j, k;
  float sum;

  #pragma omp parallel
  #pragma omp single
  for (i = 0; i < size; i++) {
    {
    long long int TM17[70];
    TM17[0] = size * 4;
    TM17[1] = TM17[0] < 0;
    TM17[2] = (TM17[1] ? TM17[0] : 0);
    TM17[3] = TM17[0] < TM17[2];
    TM17[4] = (TM17[3] ? TM17[0] : TM17[2]);
    TM17[5] = 0 < TM17[4];
    TM17[6] = (TM17[5] ? 0 : TM17[4]);
    TM17[7] = 0 < TM17[6];
    TM17[8] = (TM17[7] ? 0 : TM17[6]);
    TM17[9] = TM17[0] < TM17[8];
    TM17[10] = (TM17[9] ? TM17[0] : TM17[8]);
    TM17[11] = TM17[10] / 4;
    TM17[12] = (TM17[11] > 0);
    TM17[13] = (TM17[12] ? TM17[11] : 0);
    TM17[14] = size + 1;
    TM17[15] = size > 0;
    TM17[16] = (TM17[15] ? size : 0);
    TM17[17] = TM17[14] * TM17[16];
    TM17[18] = size + TM17[17];
    TM17[19] = size + -1;
    TM17[20] = -1 * TM17[16];
    TM17[21] = TM17[19] + TM17[20];
    TM17[22] = size * TM17[21];
    TM17[23] = TM17[18] + TM17[22];
    TM17[24] = TM17[23] * 4;
    TM17[25] = TM17[17] * 4;
    TM17[26] = size * TM17[16];
    TM17[27] = TM17[16] + TM17[26];
    TM17[28] = TM17[27] * 4;
    TM17[29] = size + TM17[26];
    TM17[30] = TM17[29] + TM17[22];
    TM17[31] = TM17[30] + TM17[16];
    TM17[32] = TM17[31] * 4;
    TM17[33] = size > TM17[16];
    TM17[34] = (TM17[33] ? size : TM17[16]);
    TM17[35] = TM17[34] + TM17[20];
    TM17[36] = TM17[17] + TM17[35];
    TM17[37] = TM17[36] * 4;
    TM17[38] = TM17[16] + TM17[35];
    TM17[39] = TM17[38] + TM17[26];
    TM17[40] = TM17[39] * 4;
    TM17[41] = TM17[26] + TM17[16];
    TM17[42] = TM17[41] * 4;
    TM17[43] = TM17[42] > TM17[37];
    TM17[44] = (TM17[43] ? TM17[42] : TM17[37]);
    TM17[45] = TM17[40] > TM17[44];
    TM17[46] = (TM17[45] ? TM17[40] : TM17[44]);
    TM17[47] = TM17[37] > TM17[46];
    TM17[48] = (TM17[47] ? TM17[37] : TM17[46]);
    TM17[49] = TM17[24] > TM17[48];
    TM17[50] = (TM17[49] ? TM17[24] : TM17[48]);
    TM17[51] = TM17[32] > TM17[50];
    TM17[52] = (TM17[51] ? TM17[32] : TM17[50]);
    TM17[53] = TM17[28] > TM17[52];
    TM17[54] = (TM17[53] ? TM17[28] : TM17[52]);
    TM17[55] = TM17[25] > TM17[54];
    TM17[56] = (TM17[55] ? TM17[25] : TM17[54]);
    TM17[57] = TM17[24] > TM17[56];
    TM17[58] = (TM17[57] ? TM17[24] : TM17[56]);
    TM17[59] = (long long int) TM17[58];
    TM17[60] = TM17[59] + 4;
    TM17[61] = TM17[60] / 4;
    TM17[62] = (TM17[61] > 0);
    TM17[63] = (TM17[62] ? TM17[61] : 0);
    TM17[64] = TM17[63] - TM17[13];
    TM17[65] = (TM17[64] > 0);
    TM17[66] = TM17[13] + TM17[64];
    TM17[67] = -1 * TM17[64];
    TM17[68] = TM17[65] ? TM17[13] : TM17[66];
    TM17[69] = TM17[65] ? TM17[64] : TM17[67];
    int tmc3 = TM18[1] * (21);
    int tmc5 = TM19[1] * (21);
    int tmc2 = 10 * (21 + tmc3);
    int tmc4 = 10 * (27 + tmc5);
    int tm_cost1 = (12 + tmc2 + tmc4);
    #pragma omp task depend(inout: a[TM17[68]:TM17[69]]) if(tm_cost1 > 41)
    {
    for (j = i; j < size; j++) {
      sum = a[i * size + j];
      for (k = 0; k < i; k++)
        sum -= a[i * size + k] * a[k * size + j];
      a[i * size + j] = sum;
    }

    for (j = i + 1; j < size; j++) {
      sum = a[j * size + i];
      for (k = 0; k < i; k++)
        sum -= a[j * size + k] * a[k * size + i];
      a[j * size + i] = sum / a[i * size + i];
    }
  }
  }
  }
}

void lud_omp_gpu(float *a, int size) {
  int i, j, k;
  float sum;

  {
    #pragma omp parallel
    #pragma omp single
    for (i = 0; i < size; i++) {
      {
      long long int TM17[70];
      TM17[0] = size * 4;
      TM17[1] = TM17[0] < 0;
      TM17[2] = (TM17[1] ? TM17[0] : 0);
      TM17[3] = TM17[0] < TM17[2];
      TM17[4] = (TM17[3] ? TM17[0] : TM17[2]);
      TM17[5] = 0 < TM17[4];
      TM17[6] = (TM17[5] ? 0 : TM17[4]);
      TM17[7] = 0 < TM17[6];
      TM17[8] = (TM17[7] ? 0 : TM17[6]);
      TM17[9] = TM17[0] < TM17[8];
      TM17[10] = (TM17[9] ? TM17[0] : TM17[8]);
      TM17[11] = TM17[10] / 4;
      TM17[12] = (TM17[11] > 0);
      TM17[13] = (TM17[12] ? TM17[11] : 0);
      TM17[14] = size + 1;
      TM17[15] = size > 0;
      TM17[16] = (TM17[15] ? size : 0);
      TM17[17] = TM17[14] * TM17[16];
      TM17[18] = size + TM17[17];
      TM17[19] = size + -1;
      TM17[20] = -1 * TM17[16];
      TM17[21] = TM17[19] + TM17[20];
      TM17[22] = size * TM17[21];
      TM17[23] = TM17[18] + TM17[22];
      TM17[24] = TM17[23] * 4;
      TM17[25] = TM17[17] * 4;
      TM17[26] = size * TM17[16];
      TM17[27] = TM17[16] + TM17[26];
      TM17[28] = TM17[27] * 4;
      TM17[29] = size + TM17[26];
      TM17[30] = TM17[29] + TM17[22];
      TM17[31] = TM17[30] + TM17[16];
      TM17[32] = TM17[31] * 4;
      TM17[33] = size > TM17[16];
      TM17[34] = (TM17[33] ? size : TM17[16]);
      TM17[35] = TM17[34] + TM17[20];
      TM17[36] = TM17[17] + TM17[35];
      TM17[37] = TM17[36] * 4;
      TM17[38] = TM17[16] + TM17[35];
      TM17[39] = TM17[38] + TM17[26];
      TM17[40] = TM17[39] * 4;
      TM17[41] = TM17[26] + TM17[16];
      TM17[42] = TM17[41] * 4;
      TM17[43] = TM17[42] > TM17[37];
      TM17[44] = (TM17[43] ? TM17[42] : TM17[37]);
      TM17[45] = TM17[40] > TM17[44];
      TM17[46] = (TM17[45] ? TM17[40] : TM17[44]);
      TM17[47] = TM17[37] > TM17[46];
      TM17[48] = (TM17[47] ? TM17[37] : TM17[46]);
      TM17[49] = TM17[24] > TM17[48];
      TM17[50] = (TM17[49] ? TM17[24] : TM17[48]);
      TM17[51] = TM17[32] > TM17[50];
      TM17[52] = (TM17[51] ? TM17[32] : TM17[50]);
      TM17[53] = TM17[28] > TM17[52];
      TM17[54] = (TM17[53] ? TM17[28] : TM17[52]);
      TM17[55] = TM17[25] > TM17[54];
      TM17[56] = (TM17[55] ? TM17[25] : TM17[54]);
      TM17[57] = TM17[24] > TM17[56];
      TM17[58] = (TM17[57] ? TM17[24] : TM17[56]);
      TM17[59] = (long long int) TM17[58];
      TM17[60] = TM17[59] + 4;
      TM17[61] = TM17[60] / 4;
      TM17[62] = (TM17[61] > 0);
      TM17[63] = (TM17[62] ? TM17[61] : 0);
      TM17[64] = TM17[63] - TM17[13];
      TM17[65] = (TM17[64] > 0);
      TM17[66] = TM17[13] + TM17[64];
      TM17[67] = -1 * TM17[64];
      TM17[68] = TM17[65] ? TM17[13] : TM17[66];
      TM17[69] = TM17[65] ? TM17[64] : TM17[67];
      int tmc3 = TM18[1] * (21);
      int tmc5 = TM19[1] * (21);
      int tmc2 = 10 * (21 + tmc3);
      int tmc4 = 10 * (27 + tmc5);
      int tm_cost1 = (12 + tmc2 + tmc4);
      #pragma omp task depend(inout: a[TM17[68]:TM17[69]]) if(tm_cost1 > 41)
      {
      {
      long long int TM17[70];
      TM17[0] = size * 4;
      TM17[1] = TM17[0] < 0;
      TM17[2] = (TM17[1] ? TM17[0] : 0);
      TM17[3] = TM17[0] < TM17[2];
      TM17[4] = (TM17[3] ? TM17[0] : TM17[2]);
      TM17[5] = 0 < TM17[4];
      TM17[6] = (TM17[5] ? 0 : TM17[4]);
      TM17[7] = 0 < TM17[6];
      TM17[8] = (TM17[7] ? 0 : TM17[6]);
      TM17[9] = TM17[0] < TM17[8];
      TM17[10] = (TM17[9] ? TM17[0] : TM17[8]);
      TM17[11] = TM17[10] / 4;
      TM17[12] = (TM17[11] > 0);
      TM17[13] = (TM17[12] ? TM17[11] : 0);
      TM17[14] = size + 1;
      TM17[15] = size > 0;
      TM17[16] = (TM17[15] ? size : 0);
      TM17[17] = TM17[14] * TM17[16];
      TM17[18] = size + TM17[17];
      TM17[19] = size + -1;
      TM17[20] = -1 * TM17[16];
      TM17[21] = TM17[19] + TM17[20];
      TM17[22] = size * TM17[21];
      TM17[23] = TM17[18] + TM17[22];
      TM17[24] = TM17[23] * 4;
      TM17[25] = TM17[17] * 4;
      TM17[26] = size * TM17[16];
      TM17[27] = TM17[16] + TM17[26];
      TM17[28] = TM17[27] * 4;
      TM17[29] = size + TM17[26];
      TM17[30] = TM17[29] + TM17[22];
      TM17[31] = TM17[30] + TM17[16];
      TM17[32] = TM17[31] * 4;
      TM17[33] = size > TM17[16];
      TM17[34] = (TM17[33] ? size : TM17[16]);
      TM17[35] = TM17[34] + TM17[20];
      TM17[36] = TM17[17] + TM17[35];
      TM17[37] = TM17[36] * 4;
      TM17[38] = TM17[16] + TM17[35];
      TM17[39] = TM17[38] + TM17[26];
      TM17[40] = TM17[39] * 4;
      TM17[41] = TM17[26] + TM17[16];
      TM17[42] = TM17[41] * 4;
      TM17[43] = TM17[42] > TM17[37];
      TM17[44] = (TM17[43] ? TM17[42] : TM17[37]);
      TM17[45] = TM17[40] > TM17[44];
      TM17[46] = (TM17[45] ? TM17[40] : TM17[44]);
      TM17[47] = TM17[37] > TM17[46];
      TM17[48] = (TM17[47] ? TM17[37] : TM17[46]);
      TM17[49] = TM17[24] > TM17[48];
      TM17[50] = (TM17[49] ? TM17[24] : TM17[48]);
      TM17[51] = TM17[32] > TM17[50];
      TM17[52] = (TM17[51] ? TM17[32] : TM17[50]);
      TM17[53] = TM17[28] > TM17[52];
      TM17[54] = (TM17[53] ? TM17[28] : TM17[52]);
      TM17[55] = TM17[25] > TM17[54];
      TM17[56] = (TM17[55] ? TM17[25] : TM17[54]);
      TM17[57] = TM17[24] > TM17[56];
      TM17[58] = (TM17[57] ? TM17[24] : TM17[56]);
      TM17[59] = (long long int) TM17[58];
      TM17[60] = TM17[59] + 4;
      TM17[61] = TM17[60] / 4;
      TM17[62] = (TM17[61] > 0);
      TM17[63] = (TM17[62] ? TM17[61] : 0);
      TM17[64] = TM17[63] - TM17[13];
      TM17[65] = (TM17[64] > 0);
      TM17[66] = TM17[13] + TM17[64];
      TM17[67] = -1 * TM17[64];
      TM17[68] = TM17[65] ? TM17[13] : TM17[66];
      TM17[69] = TM17[65] ? TM17[64] : TM17[67];
      int tmc3 = TM18[1] * (27);
      int tmc5 = TM19[1] * (27);
      int tmc2 = 10 * (21 + tmc3);
      int tmc4 = 10 * (27 + tmc5);
      int tm_cost1 = (12 + tmc2 + tmc4);
      #pragma omp task depend(inout: a[TM17[68]:TM17[69]]) if(tm_cost1 > 41)
      {
      for (j = i; j < size; j++) {
        sum = a[i * size + j];
        for (k = 0; k < i; k++)
          sum -= a[i * size + k] * a[k * size + j];
        a[i * size + j] = sum;
      }

      for (j = i + 1; j < size; j++) {
        sum = a[j * size + i];
        for (k = 0; k < i; k++)
          sum -= a[j * size + k] * a[k * size + i];
        a[j * size + i] = sum / a[i * size + i];
      }
    }
    }
    }
  }
}

