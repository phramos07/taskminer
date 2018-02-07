/*
    This program makes the decomposition of matrices.
    It receives an input array and returns two triangular matrices in the same array b.
    This program create a csv file with the time execution results for each function(CPU,GPU) in this format: size of matrix, cpu time, gpu time.
    
    Author: Gleison Souza Diniz Mendonça 
    Date: 04-01-2015
    version 1.0
    
    Run:
    folder_ipmacc/ipmacc folder_archive/LU_decomposition.c
    ./a.out
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include "../../common/mgbenchUtilFunctions.h"


#define GPU_DEVICE 0
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05
#define SIZE 500
#define points 250
#define var SIZE/points


// Initialize matrices.
void init(int s, float *a, float *b) {
    int i, j,q;
    q = s * s;
    long long int AI1[6];
    AI1[0] = s > 0;
    AI1[1] = (AI1[0] ? s : 0);
    AI1[2] = s * AI1[1];
    AI1[3] = AI1[2] + s;
    AI1[4] = AI1[3] * 4;
    AI1[5] = AI1[4] / 4;
    #pragma acc data copy(b[0:AI1[5]],a[0:AI1[5]])
    #pragma acc kernels
    #pragma acc loop independent
    for (i = 0; i < s; ++i) 
    {
        for (j = 0; j < s; ++j)
        {
            a[i * s + j] = (float)(q-(10*i + 5*j));
            b[i * s + j] = 0.0f;
        }
    }
}

/// Crout algorithm GPU
/// s = size of matrix
void Crout_GPU(int s, float *a, float *b){
    int k,j,i;
    float sum;

    #pragma omp target device (GPU_DEVICE)
    #pragma omp target map(to: a[0:SIZE*SIZE]) map(tofrom: b[0:SIZE*SIZE])
    {
        #pragma omp parallel for
        long long int AI1[53];
        AI1[0] = s + 1;
        AI1[1] = s > 0;
        AI1[2] = (AI1[1] ? s : 0);
        AI1[3] = AI1[0] * AI1[2];
        AI1[4] = 1 + AI1[3];
        AI1[5] = s + -1;
        AI1[6] = -1 * AI1[2];
        AI1[7] = AI1[5] + AI1[6];
        AI1[8] = AI1[4] + AI1[7];
        AI1[9] = AI1[8] * 4;
        AI1[10] = s > AI1[2];
        AI1[11] = (AI1[10] ? s : AI1[2]);
        AI1[12] = AI1[11] + AI1[6];
        AI1[13] = s * AI1[12];
        AI1[14] = AI1[3] + AI1[13];
        AI1[15] = AI1[14] * 4;
        AI1[16] = AI1[9] > AI1[15];
        AI1[17] = (AI1[16] ? AI1[9] : AI1[15]);
        AI1[18] = AI1[17] / 4;
        AI1[19] = AI1[0] * 4;
        AI1[20] = AI1[19] < 0;
        AI1[21] = (AI1[20] ? AI1[19] : 0);
        AI1[22] = 0 < AI1[21];
        AI1[23] = (AI1[22] ? 0 : AI1[21]);
        AI1[24] = 4 < AI1[23];
        AI1[25] = (AI1[24] ? 4 : AI1[23]);
        AI1[26] = AI1[25] / 4;
        AI1[27] = AI1[3] * 4;
        AI1[28] = AI1[0] + AI1[3];
        AI1[29] = AI1[0] * AI1[7];
        AI1[30] = AI1[28] + AI1[29];
        AI1[31] = AI1[30] * 4;
        AI1[32] = s * AI1[2];
        AI1[33] = AI1[32] + AI1[2];
        AI1[34] = AI1[33] * 4;
        AI1[35] = AI1[2] + AI1[32];
        AI1[36] = AI1[35] * 4;
        AI1[37] = AI1[32] + AI1[13];
        AI1[38] = AI1[37] + AI1[2];
        AI1[39] = AI1[38] * 4;
        AI1[40] = AI1[36] > AI1[39];
        AI1[41] = (AI1[40] ? AI1[36] : AI1[39]);
        AI1[42] = AI1[15] > AI1[41];
        AI1[43] = (AI1[42] ? AI1[15] : AI1[41]);
        AI1[44] = AI1[34] > AI1[43];
        AI1[45] = (AI1[44] ? AI1[34] : AI1[43]);
        AI1[46] = AI1[31] > AI1[45];
        AI1[47] = (AI1[46] ? AI1[31] : AI1[45]);
        AI1[48] = AI1[27] > AI1[47];
        AI1[49] = (AI1[48] ? AI1[27] : AI1[47]);
        AI1[50] = AI1[9] > AI1[49];
        AI1[51] = (AI1[50] ? AI1[9] : AI1[49]);
        AI1[52] = AI1[51] / 4;
        #pragma acc data copy(a[0:AI1[18]],b[AI1[26]:AI1[52]])
        #pragma acc kernels
        #pragma acc loop independent
        for(k=0;k<s;++k)
        {
            for(j=k;j<s;++j)
            {
                sum=0.0;
                for(i=0;i<k;++i)
                {
                    sum+=b[j*s+i]*b[i*s+k];
                }
                b[j*s+k]=(a[j*s+k]-sum); // not dividing by diagonals
            }
            for(i=k+1;i<s;++i)
            {
                sum=0.0;
                for(j=0;j<k;++j)
                {
                    sum+=b[k*s+j]*b[i*s+i];
                }
                b[k*s+i]=(a[k*s+i]-sum)/b[k*s+k];
            }
        }
    }
}

void Crout_CPU(int s, float *a, float *b){
    int k,j,i;
    float sum;

    long long int AI1[53];
    AI1[0] = s + 1;
    AI1[1] = s > 0;
    AI1[2] = (AI1[1] ? s : 0);
    AI1[3] = AI1[0] * AI1[2];
    AI1[4] = 1 + AI1[3];
    AI1[5] = s + -1;
    AI1[6] = -1 * AI1[2];
    AI1[7] = AI1[5] + AI1[6];
    AI1[8] = AI1[4] + AI1[7];
    AI1[9] = AI1[8] * 4;
    AI1[10] = s > AI1[2];
    AI1[11] = (AI1[10] ? s : AI1[2]);
    AI1[12] = AI1[11] + AI1[6];
    AI1[13] = s * AI1[12];
    AI1[14] = AI1[3] + AI1[13];
    AI1[15] = AI1[14] * 4;
    AI1[16] = AI1[9] > AI1[15];
    AI1[17] = (AI1[16] ? AI1[9] : AI1[15]);
    AI1[18] = AI1[17] / 4;
    AI1[19] = AI1[0] * 4;
    AI1[20] = AI1[19] < 0;
    AI1[21] = (AI1[20] ? AI1[19] : 0);
    AI1[22] = 0 < AI1[21];
    AI1[23] = (AI1[22] ? 0 : AI1[21]);
    AI1[24] = 4 < AI1[23];
    AI1[25] = (AI1[24] ? 4 : AI1[23]);
    AI1[26] = AI1[25] / 4;
    AI1[27] = AI1[3] * 4;
    AI1[28] = AI1[0] + AI1[3];
    AI1[29] = AI1[0] * AI1[7];
    AI1[30] = AI1[28] + AI1[29];
    AI1[31] = AI1[30] * 4;
    AI1[32] = s * AI1[2];
    AI1[33] = AI1[32] + AI1[2];
    AI1[34] = AI1[33] * 4;
    AI1[35] = AI1[2] + AI1[32];
    AI1[36] = AI1[35] * 4;
    AI1[37] = AI1[32] + AI1[13];
    AI1[38] = AI1[37] + AI1[2];
    AI1[39] = AI1[38] * 4;
    AI1[40] = AI1[36] > AI1[39];
    AI1[41] = (AI1[40] ? AI1[36] : AI1[39]);
    AI1[42] = AI1[15] > AI1[41];
    AI1[43] = (AI1[42] ? AI1[15] : AI1[41]);
    AI1[44] = AI1[34] > AI1[43];
    AI1[45] = (AI1[44] ? AI1[34] : AI1[43]);
    AI1[46] = AI1[31] > AI1[45];
    AI1[47] = (AI1[46] ? AI1[31] : AI1[45]);
    AI1[48] = AI1[27] > AI1[47];
    AI1[49] = (AI1[48] ? AI1[27] : AI1[47]);
    AI1[50] = AI1[9] > AI1[49];
    AI1[51] = (AI1[50] ? AI1[9] : AI1[49]);
    AI1[52] = AI1[51] / 4;
    #pragma acc data copy(a[0:AI1[18]],b[AI1[26]:AI1[52]])
    #pragma acc kernels
    #pragma acc loop independent
    for(k=0;k<s;++k)
    {
        for(j=k;j<s;++j)
        {
            sum=0.0;
            for(i=0;i<k;++i)
            {
                sum+=b[j*s+i]*b[i*s+k];
            }
            b[j*s+k]=(a[j*s+k]-sum); // not dividing by diagonals
        }
        for(i=k+1;i<s;++i)
        {
            sum=0.0;
            for(j=0;j<k;++j)
            {
                sum+=b[k*s+j]*b[i*s+i];
            }
            b[k*s+i]=(a[k*s+i]-sum)/b[k*s+k];
        }
    }
}

void compareResults(float *b_cpu, float *b_gpu)
{
  int i, j, fail;
  fail = 0;
	
  #pragma acc data copy(b_cpu[0:250500],b_gpu[0:250500])
  #pragma acc kernels
  #pragma acc loop independent
  for (i=0; i < SIZE; i++) 
    {
      for (j=0; j < SIZE; j++) 
	{
	  if (percentDiff(b_cpu[i*SIZE + j], b_gpu[i*SIZE + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
	    {
	      fail++;
	    }
	}
    }
	
  // Print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

int main(int argc, char *argv[]) 
{
    double t_start, t_end;
    int i;

    float *a, *b_cpu, *b_gpu;
    a = (float *) malloc(sizeof(float) * SIZE * SIZE);
    b_cpu = (float *) malloc(sizeof(float) * SIZE * SIZE);
    b_gpu = (float *) malloc(sizeof(float) * SIZE * SIZE);
    
    fprintf(stdout,"<< LU decomposition GPU >>\n");
 
    t_start = rtclock();
    for(i=2;i<SIZE;i+=var)
    {
        init(i, a, b_gpu);
        Crout_GPU(i, a, b_gpu);  
    }
    t_end = rtclock();
    fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);	

    t_start = rtclock();
    for(i=2;i<SIZE;i+=var)
    {
        init(i, a, b_cpu);
        Crout_CPU(i, a, b_cpu);
    }
    t_end = rtclock();
    fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);	

    compareResults(b_cpu, b_gpu);

    free(a);
    free(b_cpu);
    free(b_gpu);

    return 0;
}





