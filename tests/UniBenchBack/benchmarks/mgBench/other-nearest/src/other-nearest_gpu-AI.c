//<libmptogpu> Error executing kernel. Global Work Size is NULL or exceeded valid range.


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include "../../common/mgbenchUtilFunctions.h"

typedef struct point
{
    int x;
    int y;
}point;

typedef struct sel_points
{
    int position;
    float value;
}sel_points;

#define SIZE 500
#define points 250
#define var SIZE/points
#define default_v 100000.00
#define GPU_DEVICE 0
#define ERROR_THRESHOLD 0.01

void init(int s, point *vector, sel_points *selected)
{
    int i,j;
    long long int AI1[7];
    AI1[0] = s > 0;
    AI1[1] = (AI1[0] ? s : 0);
    AI1[2] = 8 * AI1[1];
    AI1[3] = 4 + AI1[2];
    AI1[4] = AI1[3] > AI1[2];
    AI1[5] = (AI1[4] ? AI1[3] : AI1[2]);
    AI1[6] = AI1[5] / 8;
    #pragma acc data copy(vector[0:AI1[6]])
    #pragma acc kernels
    #pragma acc loop independent
    for(i=0;i<s;i++)
    {
        vector[i].x = i;
        vector[i].y = i*2;
    }
    long long int AI2[10];
    AI2[0] = s > 0;
    AI2[1] = (AI2[0] ? s : 0);
    AI2[2] = s * AI2[1];
    AI2[3] = AI2[2] + s;
    AI2[4] = 2 * AI2[3];
    AI2[5] = AI2[4] * 4;
    AI2[6] = AI2[3] * 8;
    AI2[7] = AI2[5] > AI2[6];
    AI2[8] = (AI2[7] ? AI2[5] : AI2[6]);
    AI2[9] = AI2[8] / 8;
    #pragma acc data copy(selected[0:AI2[9]])
    #pragma acc kernels
    #pragma acc loop independent
    for(i=0;i<s;i++)
    {
        for(j=0;j<s;j++)
        {
            selected[i*s+j].position = 0;
            selected[i*s+j].value = default_v;
        }
    }
}

void k_nearest_gpu(int s, point *vector, sel_points *selected)
{
    int i,j,m,q;
    q = s*s;

    #pragma omp target device (GPU_DEVICE)
    #pragma omp target map(to: vector[0: s]) map(tofrom: selected[0:q])
    {
	#pragma omp parallel for collapse(2) 
        long long int AI1[47];
        AI1[0] = s > 0;
        AI1[1] = (AI1[0] ? s : 0);
        AI1[2] = 8 * AI1[1];
        AI1[3] = 12 + AI1[2];
        AI1[4] = s + -1;
        AI1[5] = -1 * AI1[1];
        AI1[6] = AI1[4] + AI1[5];
        AI1[7] = 8 * AI1[6];
        AI1[8] = AI1[3] + AI1[7];
        AI1[9] = 4 + AI1[2];
        AI1[10] = 8 + AI1[2];
        AI1[11] = AI1[10] + AI1[7];
        AI1[12] = AI1[11] > AI1[2];
        AI1[13] = (AI1[12] ? AI1[11] : AI1[2]);
        AI1[14] = AI1[9] > AI1[13];
        AI1[15] = (AI1[14] ? AI1[9] : AI1[13]);
        AI1[16] = AI1[8] > AI1[15];
        AI1[17] = (AI1[16] ? AI1[8] : AI1[15]);
        AI1[18] = AI1[17] / 8;
        AI1[19] = s * 8;
        AI1[20] = 2 * s;
        AI1[21] = AI1[20] * 4;
        AI1[22] = AI1[21] < 8;
        AI1[23] = (AI1[22] ? AI1[21] : 8);
        AI1[24] = AI1[19] < AI1[23];
        AI1[25] = (AI1[24] ? AI1[19] : AI1[23]);
        AI1[26] = AI1[25] / 8;
        AI1[27] = s + 1;
        AI1[28] = AI1[27] * AI1[1];
        AI1[29] = s + AI1[28];
        AI1[30] = s * AI1[6];
        AI1[31] = AI1[29] + AI1[30];
        AI1[32] = AI1[31] * 8;
        AI1[33] = 2 * AI1[31];
        AI1[34] = AI1[33] * 4;
        AI1[35] = 1 + AI1[28];
        AI1[36] = AI1[35] + AI1[6];
        AI1[37] = AI1[36] * 8;
        AI1[38] = 2 * AI1[36];
        AI1[39] = AI1[38] * 4;
        AI1[40] = AI1[37] > AI1[39];
        AI1[41] = (AI1[40] ? AI1[37] : AI1[39]);
        AI1[42] = AI1[34] > AI1[41];
        AI1[43] = (AI1[42] ? AI1[34] : AI1[41]);
        AI1[44] = AI1[32] > AI1[43];
        AI1[45] = (AI1[44] ? AI1[32] : AI1[43]);
        AI1[46] = AI1[45] / 8;
        #pragma acc data copy(vector[0:AI1[18]],selected[AI1[26]:AI1[46]])
        #pragma acc kernels
        #pragma acc loop independent
        for(i=0;i<s;i++)
        {
            for(j=i+1;j<s;j++)
            {
                float distance,x,y;
                x = vector[i].x - vector[j].x;
                y = vector[i].y - vector[j].y;
                x = x * x;
                y = y * y;
                
                distance = x + y;
                distance = sqrt(distance);
                
                selected[i*s+j].value = distance;
                selected[i*s+j].position = j;
                
                selected[j*s+i].value = distance;
                selected[j*s+i].position = i;
            }
        }
        
        /// for each line in matrix
        /// order values
	#pragma omp parallel for collapse(1)
    long long int AI2[17];
    AI2[0] = s > 0;
    AI2[1] = (AI2[0] ? s : 0);
    AI2[2] = s * AI2[1];
    AI2[3] = 1 + AI2[2];
    AI2[4] = AI2[3] + s;
    AI2[5] = s + -1;
    AI2[6] = -1 * s;
    AI2[7] = AI2[5] + AI2[6];
    AI2[8] = AI2[4] + AI2[7];
    AI2[9] = 2 * AI2[8];
    AI2[10] = AI2[9] * 4;
    AI2[11] = AI2[2] + s;
    AI2[12] = 2 * AI2[11];
    AI2[13] = AI2[12] * 4;
    AI2[14] = AI2[10] > AI2[13];
    AI2[15] = (AI2[14] ? AI2[10] : AI2[13]);
    AI2[16] = AI2[15] / 8;
    #pragma acc data copy(selected[0:AI2[16]])
    #pragma acc kernels
    #pragma acc loop independent
    for(i=0;i<s;i++)
        {
            for(j=0;j<s;j++)
            {
                for(m=j+1;m<s;m++)
                {
                    if(selected[i*s+j].value>selected[i*s+m].value)
                    {
                        sel_points aux;
                        aux = selected[i*s+j];
                        selected[i*s+j] = selected[i*s+m];
                        selected[i*s+m] = aux;
                    }
                } 
               
            }
        }
    }
}


void k_nearest_cpu(int s, point *vector, sel_points *selected)
{
    int i,j;
    long long int AI1[47];
    AI1[0] = s > 0;
    AI1[1] = (AI1[0] ? s : 0);
    AI1[2] = 8 * AI1[1];
    AI1[3] = 12 + AI1[2];
    AI1[4] = s + -1;
    AI1[5] = -1 * AI1[1];
    AI1[6] = AI1[4] + AI1[5];
    AI1[7] = 8 * AI1[6];
    AI1[8] = AI1[3] + AI1[7];
    AI1[9] = 4 + AI1[2];
    AI1[10] = 8 + AI1[2];
    AI1[11] = AI1[10] + AI1[7];
    AI1[12] = AI1[11] > AI1[2];
    AI1[13] = (AI1[12] ? AI1[11] : AI1[2]);
    AI1[14] = AI1[9] > AI1[13];
    AI1[15] = (AI1[14] ? AI1[9] : AI1[13]);
    AI1[16] = AI1[8] > AI1[15];
    AI1[17] = (AI1[16] ? AI1[8] : AI1[15]);
    AI1[18] = AI1[17] / 8;
    AI1[19] = s * 8;
    AI1[20] = 2 * s;
    AI1[21] = AI1[20] * 4;
    AI1[22] = AI1[21] < 8;
    AI1[23] = (AI1[22] ? AI1[21] : 8);
    AI1[24] = AI1[19] < AI1[23];
    AI1[25] = (AI1[24] ? AI1[19] : AI1[23]);
    AI1[26] = AI1[25] / 8;
    AI1[27] = s + 1;
    AI1[28] = AI1[27] * AI1[1];
    AI1[29] = s + AI1[28];
    AI1[30] = s * AI1[6];
    AI1[31] = AI1[29] + AI1[30];
    AI1[32] = AI1[31] * 8;
    AI1[33] = 2 * AI1[31];
    AI1[34] = AI1[33] * 4;
    AI1[35] = 1 + AI1[28];
    AI1[36] = AI1[35] + AI1[6];
    AI1[37] = AI1[36] * 8;
    AI1[38] = 2 * AI1[36];
    AI1[39] = AI1[38] * 4;
    AI1[40] = AI1[37] > AI1[39];
    AI1[41] = (AI1[40] ? AI1[37] : AI1[39]);
    AI1[42] = AI1[34] > AI1[41];
    AI1[43] = (AI1[42] ? AI1[34] : AI1[41]);
    AI1[44] = AI1[32] > AI1[43];
    AI1[45] = (AI1[44] ? AI1[32] : AI1[43]);
    AI1[46] = AI1[45] / 8;
    #pragma acc data copy(vector[0:AI1[18]],selected[AI1[26]:AI1[46]])
    #pragma acc kernels
    #pragma acc loop independent
    for(i=0;i<s;i++)
    {
        for(j=i+1;j<s;j++)
        {
            float distance,x,y;
            x = vector[i].x - vector[j].x;
            y = vector[i].y - vector[j].y;
            x = x * x;
            y = y * y;
            
            distance = x + y;
            distance = sqrt(distance);
            
            selected[i*s+j].value = distance;
            selected[i*s+j].position = j;
            
            selected[j*s+i].value = distance;
            selected[j*s+i].position = i;
        }
    }
}

void order_points(int s, point *vector, sel_points *selected)
{
    int i;
    long long int AI1[17];
    AI1[0] = s > 0;
    AI1[1] = (AI1[0] ? s : 0);
    AI1[2] = s * AI1[1];
    AI1[3] = 1 + AI1[2];
    AI1[4] = AI1[3] + s;
    AI1[5] = s + -1;
    AI1[6] = -1 * s;
    AI1[7] = AI1[5] + AI1[6];
    AI1[8] = AI1[4] + AI1[7];
    AI1[9] = 2 * AI1[8];
    AI1[10] = AI1[9] * 4;
    AI1[11] = AI1[2] + s;
    AI1[12] = 2 * AI1[11];
    AI1[13] = AI1[12] * 4;
    AI1[14] = AI1[10] > AI1[13];
    AI1[15] = (AI1[14] ? AI1[10] : AI1[13]);
    AI1[16] = AI1[15] / 8;
    #pragma acc data copy(selected[0:AI1[16]])
    #pragma acc kernels
    #pragma acc loop independent
    for(i=0;i<s;i++)
    {
        /// for each line in matrix
        /// order values
        int j;
        for(j=0;j<s;j++)
        {
            int m;
            for(m=j+1;m<s;m++)
            {
                if(selected[i*s+j].value>selected[i*s+m].value)
                {
                    sel_points aux;
                    aux = selected[i*s+j];
                    selected[i*s+j] = selected[i*s+m];
                    selected[i*s+m] = aux;
                }
            } 
        }
    }
}


void compareResults(sel_points* B, sel_points* B_GPU)
{
  int i, j, fail;
  fail = 0;
	
  // Compare B and B_GPU
  #pragma acc data copy(B[0:250500],B_GPU[0:250500])
  #pragma acc kernels
  #pragma acc loop independent
  for (i=0; i < SIZE; i++) 
    {
      for (j=0; j < SIZE; j++) 
	{
	  //Value
	  if (percentDiff(B[i*SIZE + j].value, B_GPU[i*SIZE + j].value) > ERROR_THRESHOLD) 
	    {
	      fail++;
	    }
	  //Position
	  if (percentDiff(B[i*SIZE + j].position, B_GPU[i*SIZE + j].position) > ERROR_THRESHOLD) 
	    {
	      fail++;
	    }
	}
    }
  // Print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", ERROR_THRESHOLD, fail);
	
}

int main(int argc, char *argv[])
{
    double t_start, t_end;
    point *vector;
    sel_points *selected_cpu, *selected_gpu;

    vector = (point *) malloc(sizeof(point) * SIZE);
    selected_cpu = (sel_points *)malloc(sizeof(sel_points) * SIZE * SIZE);   
    selected_gpu = (sel_points *)malloc(sizeof(sel_points) * SIZE * SIZE);
     
    int i;
    
    fprintf(stdout, "<< Nearest >>\n");
    
    t_start = rtclock();
    for(i=(var-1);i<SIZE;i+=var)
    {
        init(i, vector, selected_cpu);
        k_nearest_cpu(i, vector, selected_cpu);
        order_points(i, vector, selected_cpu);
    }
    t_end = rtclock();
    fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);	


    t_start = rtclock();
    for(i=(var-1);i<SIZE;i+=var)
    {
        init(i, vector, selected_gpu);
        k_nearest_gpu(i, vector, selected_gpu);
    }
    t_end = rtclock();
    fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);	

    compareResults(selected_cpu, selected_gpu);

    free(selected_cpu);
    free(selected_gpu);
    free(vector);
    return 0;
}


