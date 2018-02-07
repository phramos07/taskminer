#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <openacc.h>
#define IPMACC_MAX1(A)   (A)
#define IPMACC_MAX2(A,B) (A>B?A:B)
#define IPMACC_MAX3(A,B,C) (A>B?(A>C?A:(B>C?B:C)):(B>C?C:B))
#ifdef __cplusplus
#include "openacc_container.h"
#endif

#include <cuda.h>



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
#define var SIZE / points



 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_0(float * a,int  q,int  s,float * b);
 
void init(int s, float *a, float *b)
{
  int i, j, q;
  q = s * s;
  long long int AI1 [7];
  AI1 [0] = s + -1;
  AI1 [1] = s * AI1 [0];
  AI1 [2] = AI1 [1] + AI1 [0];
  AI1 [3] = AI1 [2] * 4;
  AI1 [4] = AI1 [3] / 4;
  AI1 [5] = (AI1 [4] > 0);
  AI1 [6] = (AI1 [5] ? AI1 [4] : 0);
    

	ipmacc_prompt((char*)"IPMACC: memory allocation b\n");
acc_present_or_create((void*)b,(AI1[6]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation a\n");
acc_present_or_create((void*)a,(AI1[6]+0)*sizeof(float ));
	ipmacc_prompt((char*)"IPMACC: memory copyin b\n");
acc_pcopyin((void*)b,(AI1[6]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin a\n");
acc_pcopyin((void*)a,(AI1[6]+0)*sizeof(float ));


{


    


/* kernel call statement [0, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 0 > gridDim: %d\tblockDim: %d\n",(((abs((int)((s))-(0+0)))/(1)))/256+(((((abs((int)((s))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_0<<<(((abs((int)((s))-(0+0)))/(1)))/256+(((((abs((int)((s))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(float *)acc_deviceptr((void*)a),
q,
s,
(float *)acc_deviceptr((void*)b));
}
/* kernel call statement*/
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Synchronizing the region with host\n");
{
cudaError err=cudaDeviceSynchronize();
if(err!=cudaSuccess){
printf("Kernel Launch Error! error code (%d)\n",err);
assert(0&&"Launch Failure!\n");}
}



}
	ipmacc_prompt((char*)"IPMACC: memory copyout b\n");
acc_copyout_and_keep((void*)b,(AI1[6]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout a\n");
acc_copyout_and_keep((void*)a,(AI1[6]+0)*sizeof(float ));



}



 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_1(float * a,float * b,int  s,float  sum);
 
void Crout_GPU(int s, float *a, float *b)
{
  int k, j, i;
  float sum;

    #pragma omp target device (GPU_DEVICE)
    #pragma omp target map(to: a[0:SIZE*SIZE]) map(tofrom: b[0:SIZE*SIZE])
  {
        #pragma omp parallel for
    long long int AI1 [26];
    AI1 [0] = s + 1;
    AI1 [1] = s + -1;
    AI1 [2] = AI1 [0] * AI1 [1];
    AI1 [3] = -1 * AI1 [1];
    AI1 [4] = AI1 [1] + AI1 [3];
    AI1 [5] = s * AI1 [4];
    AI1 [6] = AI1 [2] + AI1 [5];
    AI1 [7] = AI1 [6] * 4;
    AI1 [8] = AI1 [7] / 4;
    AI1 [9] = (AI1 [8] > 0);
    AI1 [10] = (AI1 [9] ? AI1 [8] : 0);
    AI1 [11] = -1 + AI1 [1];
    AI1 [12] = s * AI1 [11];
    AI1 [13] = AI1 [1] + AI1 [12];
    AI1 [14] = AI1 [13] * 4;
    AI1 [15] = s * AI1 [1];
    AI1 [16] = AI1 [15] + AI1 [5];
    AI1 [17] = AI1 [16] + AI1 [11];
    AI1 [18] = AI1 [17] * 4;
    AI1 [19] = AI1 [14] > AI1 [18];
    AI1 [20] = (AI1 [19] ? AI1 [14] : AI1 [18]);
    AI1 [21] = AI1 [7] > AI1 [20];
    AI1 [22] = (AI1 [21] ? AI1 [7] : AI1 [20]);
    AI1 [23] = AI1 [22] / 4;
    AI1 [24] = (AI1 [23] > 0);
    AI1 [25] = (AI1 [24] ? AI1 [23] : 0);
        

	ipmacc_prompt((char*)"IPMACC: memory allocation a\n");
acc_present_or_create((void*)a,(AI1[10]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation b\n");
acc_present_or_create((void*)b,(AI1[25]+0)*sizeof(float ));
	ipmacc_prompt((char*)"IPMACC: memory copyin a\n");
acc_pcopyin((void*)a,(AI1[10]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin b\n");
acc_pcopyin((void*)b,(AI1[25]+0)*sizeof(float ));


{


        


/* kernel call statement [1, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 1 > gridDim: %d\tblockDim: %d\n",(((abs((int)((s))-(0+0)))/(1)))/256+(((((abs((int)((s))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_1<<<(((abs((int)((s))-(0+0)))/(1)))/256+(((((abs((int)((s))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(float *)acc_deviceptr((void*)a),
(float *)acc_deviceptr((void*)b),
s,
sum);
}
/* kernel call statement*/
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Synchronizing the region with host\n");
{
cudaError err=cudaDeviceSynchronize();
if(err!=cudaSuccess){
printf("Kernel Launch Error! error code (%d)\n",err);
assert(0&&"Launch Failure!\n");}
}



}
	ipmacc_prompt((char*)"IPMACC: memory copyout a\n");
acc_copyout_and_keep((void*)a,(AI1[10]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout b\n");
acc_copyout_and_keep((void*)b,(AI1[25]+0)*sizeof(float ));



  }
}

 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_2(float * a,float * b,int  s,float  sum);
 
void Crout_CPU(int s, float *a, float *b)
{
  int k, j, i;
  float sum;

  long long int AI1 [26];
  AI1 [0] = s + 1;
  AI1 [1] = s + -1;
  AI1 [2] = AI1 [0] * AI1 [1];
  AI1 [3] = -1 * AI1 [1];
  AI1 [4] = AI1 [1] + AI1 [3];
  AI1 [5] = s * AI1 [4];
  AI1 [6] = AI1 [2] + AI1 [5];
  AI1 [7] = AI1 [6] * 4;
  AI1 [8] = AI1 [7] / 4;
  AI1 [9] = (AI1 [8] > 0);
  AI1 [10] = (AI1 [9] ? AI1 [8] : 0);
  AI1 [11] = -1 + AI1 [1];
  AI1 [12] = s * AI1 [11];
  AI1 [13] = AI1 [1] + AI1 [12];
  AI1 [14] = AI1 [13] * 4;
  AI1 [15] = s * AI1 [1];
  AI1 [16] = AI1 [15] + AI1 [5];
  AI1 [17] = AI1 [16] + AI1 [11];
  AI1 [18] = AI1 [17] * 4;
  AI1 [19] = AI1 [14] > AI1 [18];
  AI1 [20] = (AI1 [19] ? AI1 [14] : AI1 [18]);
  AI1 [21] = AI1 [7] > AI1 [20];
  AI1 [22] = (AI1 [21] ? AI1 [7] : AI1 [20]);
  AI1 [23] = AI1 [22] / 4;
  AI1 [24] = (AI1 [23] > 0);
  AI1 [25] = (AI1 [24] ? AI1 [23] : 0);
    

	ipmacc_prompt((char*)"IPMACC: memory allocation a\n");
acc_present_or_create((void*)a,(AI1[10]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation b\n");
acc_present_or_create((void*)b,(AI1[25]+0)*sizeof(float ));
	ipmacc_prompt((char*)"IPMACC: memory copyin a\n");
acc_pcopyin((void*)a,(AI1[10]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin b\n");
acc_pcopyin((void*)b,(AI1[25]+0)*sizeof(float ));


{


    


/* kernel call statement [2, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 2 > gridDim: %d\tblockDim: %d\n",(((abs((int)((s))-(0+0)))/(1)))/256+(((((abs((int)((s))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_2<<<(((abs((int)((s))-(0+0)))/(1)))/256+(((((abs((int)((s))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(float *)acc_deviceptr((void*)a),
(float *)acc_deviceptr((void*)b),
s,
sum);
}
/* kernel call statement*/
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Synchronizing the region with host\n");
{
cudaError err=cudaDeviceSynchronize();
if(err!=cudaSuccess){
printf("Kernel Launch Error! error code (%d)\n",err);
assert(0&&"Launch Failure!\n");}
}



}
	ipmacc_prompt((char*)"IPMACC: memory copyout a\n");
acc_copyout_and_keep((void*)a,(AI1[10]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout b\n");
acc_copyout_and_keep((void*)b,(AI1[25]+0)*sizeof(float ));



}

 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_3(float * b_gpu,float * b_cpu,int  fail);
 
void compareResults(float *b_cpu, float *b_gpu)
{
  int i, j, fail;
  fail = 0;

  

	ipmacc_prompt((char*)"IPMACC: memory allocation b_cpu\n");
acc_present_or_create((void*)b_cpu,(249999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation b_gpu\n");
acc_present_or_create((void*)b_gpu,(249999+0)*sizeof(float ));
	ipmacc_prompt((char*)"IPMACC: memory copyin b_cpu\n");
acc_pcopyin((void*)b_cpu,(249999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin b_gpu\n");
acc_pcopyin((void*)b_gpu,(249999+0)*sizeof(float ));


{


  


/* kernel call statement [3, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 3 > gridDim: %d\tblockDim: %d\n",(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_3<<<(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(float *)acc_deviceptr((void*)b_gpu),
(float *)acc_deviceptr((void*)b_cpu),
fail);
}
/* kernel call statement*/
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Synchronizing the region with host\n");
{
cudaError err=cudaDeviceSynchronize();
if(err!=cudaSuccess){
printf("Kernel Launch Error! error code (%d)\n",err);
assert(0&&"Launch Failure!\n");}
}



}
	ipmacc_prompt((char*)"IPMACC: memory copyout b_cpu\n");
acc_copyout_and_keep((void*)b_cpu,(249999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout b_gpu\n");
acc_copyout_and_keep((void*)b_gpu,(249999+0)*sizeof(float ));




  
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

int main(int argc, char *argv[])
{
  double t_start, t_end;
  int i;

  float *a, *b_cpu, *b_gpu;
  a = (float*)malloc(sizeof(float) * SIZE * SIZE);
  b_cpu = (float*)malloc(sizeof(float) * SIZE * SIZE);
  b_gpu = (float*)malloc(sizeof(float) * SIZE * SIZE);

  fprintf(stdout, "<< LU decomposition GPU >>\n");

  t_start = rtclock();
  for (i = 2; i < SIZE; i += var) {
    init(i, a, b_gpu);
    Crout_GPU(i, a, b_gpu);
  }
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  for (i = 2; i < SIZE; i += var) {
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


__device__ float __accelerator_absVal( float a ) {
  if ( a  < 0 ) {
   return  ( a  * -1) ; 
 } else
  {
   return  a ; 
 } 
}
__device__ float __accelerator_percentDiff( double val1 , double val2 ) {
  if  ( ( __accelerator_absVal( val1 )  < 0.01) && ( __accelerator_absVal( val2 )  < 0.01) ) {
   return  0.0f ; 
 } else
  {
       return  100.0f * ( __accelerator_absVal( __accelerator_absVal( val1  - val2 )  / __accelerator_absVal( val1  + 0.00000001f ) ) ) ; 
 } 
}

 __global__ void __generated_kernel_region_0(float * a,int  q,int  s,float * b){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  j;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < s)
{
for(j = 0; j < s; ++j)
{
      a [i * s + j] = (float)(q - (10 * i + 5 * j));
      b [i * s + j] = 0.0f;
    }
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_1(float * a,float * b,int  s,float  sum){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  k;
int  j;
{
{
{
 k=0+(__kernel_getuid_x);
if( k < s)
{
for(j = k; j < s; ++j)
{
        sum = 0.0;
for(i = 0; i < k; ++i)
{
          sum += b [j * s + i] * b [i * s + k];
        }
b [j * s + k] = (a [j * s + k] - sum); 
      }

for(i = k + 1; i < s; ++i)
{
        sum = 0.0;
for(j = 0; j < k; ++j)
{
          sum += b [k * s + j] * b [i * s + i];
        }
b [k * s + i] = (a [k * s + i] - sum) / b [k * s + k];
      }
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_2(float * a,float * b,int  s,float  sum){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  k;
int  j;
{
{
{
 k=0+(__kernel_getuid_x);
if( k < s)
{
for(j = k; j < s; ++j)
{
      sum = 0.0;
for(i = 0; i < k; ++i)
{
        sum += b [j * s + i] * b [i * s + k];
      }
b [j * s + k] = (a [j * s + k] - sum); 
    }

for(i = k + 1; i < s; ++i)
{
      sum = 0.0;
for(j = 0; j < k; ++j)
{
        sum += b [k * s + j] * b [i * s + i];
      }
b [k * s + i] = (a [k * s + i] - sum) / b [k * s + k];
    }
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_3(float * b_gpu,float * b_cpu,int  fail){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  j;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < SIZE)
{
for(j = 0; j < SIZE; j++)
{
      if (__accelerator_percentDiff(b_cpu [i * SIZE + j], b_gpu [i * SIZE + j]) > PERCENT_DIFF_ERROR_THRESHOLD) {
        fail++;
      }
    }
}

}
}
}
//append writeback of scalar variables
}

