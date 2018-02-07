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
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include "../../common/mgbenchUtilFunctions.h"

#define SIZE 1000
#define GPU_DEVICE 0
#define ERROR_THRESHOLD 0.05



 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_0(float * A,float * B_GPU,int  q,float * B_CPU);
 
void init_arrays(float *A, float *B_GPU, float *B_CPU)
{
  int i, j, q;
  q = SIZE * SIZE;

    

	ipmacc_prompt((char*)"IPMACC: memory allocation B_GPU\n");
acc_present_or_create((void*)B_GPU,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation B_CPU\n");
acc_present_or_create((void*)B_CPU,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation A\n");
acc_present_or_create((void*)A,(999999+0)*sizeof(float ));
	ipmacc_prompt((char*)"IPMACC: memory copyin B_GPU\n");
acc_pcopyin((void*)B_GPU,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin B_CPU\n");
acc_pcopyin((void*)B_CPU,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin A\n");
acc_pcopyin((void*)A,(999999+0)*sizeof(float ));


{


    


/* kernel call statement [0, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 0 > gridDim: %d\tblockDim: %d\n",(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_0<<<(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(float *)acc_deviceptr((void*)A),
(float *)acc_deviceptr((void*)B_GPU),
q,
(float *)acc_deviceptr((void*)B_CPU));
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
	ipmacc_prompt((char*)"IPMACC: memory copyout B_GPU\n");
acc_copyout_and_keep((void*)B_GPU,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout B_CPU\n");
acc_copyout_and_keep((void*)B_CPU,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout A\n");
acc_copyout_and_keep((void*)A,(999999+0)*sizeof(float ));



}



void cholesky_GPU(float *A, float *B)
{
  int i, j, k, l;

  float t;

    #pragma omp target device(GPU_DEVICE)
    #pragma omp target map(to: A[0:SIZE*SIZE]) map(tofrom: B[0:SIZE*SIZE])
  {
  #pragma omp parallel for collapse(1)
    for (i = 0; i < SIZE; i++) {
      for (j = 0; j <= i; j++) {
        t = 0.0f;
        for (k = 0; k < j; k++) {
          if (B [i * SIZE + k] != 0.0f && B [j * SIZE + k] != 0.0f) {
            t += B [i * SIZE + k] * B [j * SIZE + k];
          }else  {
            k--;
          }
        }
        if (i == j) {
          B [i * SIZE + j] = sqrt((A [i * SIZE + i] - t));
        }else  {
          if (B [j * SIZE + j] != 0.0f) {
            B [i * SIZE + j] = (1.0 / B [j * SIZE + j] * (A [i * SIZE + j] - t));
          }else  {
            j--;
          }
        }
      }
    }
  }
}

 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_1(float * A,float * B);
 
void cholesky_CPU(float *A, float *B)
{
  int i, j, k;

    

	ipmacc_prompt((char*)"IPMACC: memory allocation A\n");
acc_present_or_create((void*)A,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation B\n");
acc_present_or_create((void*)B,(999999+0)*sizeof(float ));
	ipmacc_prompt((char*)"IPMACC: memory copyin A\n");
acc_pcopyin((void*)A,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin B\n");
acc_pcopyin((void*)B,(999999+0)*sizeof(float ));


{


    


/* kernel call statement [1, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 1 > gridDim: %d\tblockDim: %d\n",(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_1<<<(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(float *)acc_deviceptr((void*)A),
(float *)acc_deviceptr((void*)B));
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
	ipmacc_prompt((char*)"IPMACC: memory copyout A\n");
acc_copyout_and_keep((void*)A,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout B\n");
acc_copyout_and_keep((void*)B,(999999+0)*sizeof(float ));



}

 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_2(float * E,float * E_GPU,int  fail);
 
void compareResults(float *E, float *E_GPU)
{
  int i, j, fail;
  fail = 0;

  

	ipmacc_prompt((char*)"IPMACC: memory allocation E\n");
acc_present_or_create((void*)E,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation E_GPU\n");
acc_present_or_create((void*)E_GPU,(999999+0)*sizeof(float ));
	ipmacc_prompt((char*)"IPMACC: memory copyin E\n");
acc_pcopyin((void*)E,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin E_GPU\n");
acc_pcopyin((void*)E_GPU,(999999+0)*sizeof(float ));


{


  


/* kernel call statement [2, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 2 > gridDim: %d\tblockDim: %d\n",(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_2<<<(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(float *)acc_deviceptr((void*)E),
(float *)acc_deviceptr((void*)E_GPU),
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
	ipmacc_prompt((char*)"IPMACC: memory copyout E\n");
acc_copyout_and_keep((void*)E,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout E_GPU\n");
acc_copyout_and_keep((void*)E_GPU,(999999+0)*sizeof(float ));




  
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", ERROR_THRESHOLD, fail);
}

int main(int argc, char *argv[])
{
  double t_start, t_end;
  float *A, *B_CPU, *B_GPU;

  A = (float*)malloc(SIZE * SIZE * sizeof(float));
  B_CPU = (float*)malloc(SIZE * SIZE * sizeof(float));
  B_GPU = (float*)malloc(SIZE * SIZE * sizeof(float));

  fprintf(stdout, "<< Cholesky >>\n");

  init_arrays(A, B_CPU, B_GPU);

  t_start = rtclock();
  cholesky_GPU(A, B_GPU);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  cholesky_CPU(A, B_CPU);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(B_CPU, B_GPU);

  free(A);
  free(B_CPU);
  free(B_GPU);

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

 __global__ void __generated_kernel_region_0(float * A,float * B_GPU,int  q,float * B_CPU){
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
for(j = 0; j < SIZE; ++j)
{
      A [i * SIZE + j] = (float)(q - (10 * i) - (5 * j));
      B_GPU [i * SIZE + j] = 0.0f;
      B_CPU [i * SIZE + j] = 0.0f;
    }
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_1(float * A,float * B){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  k;
int  j;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < SIZE)
{
for(j = 0; j <= i; j++)
{
      float t;
      t = 0.0f;
for(k = 0; k < j; k++)
{
        t += B [i * SIZE + k] * B [j * SIZE + k];
      }
if (i == j) {
        B [i * SIZE + j] = sqrt((A [i * SIZE + i] - t));
      }else  {
        B [i * SIZE + j] = (1.0 / B [j * SIZE + j] * (A [i * SIZE + j] - t));
      }
    }
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_2(float * E,float * E_GPU,int  fail){
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
      if (__accelerator_percentDiff(E [i * SIZE + j], E_GPU [i * SIZE + j]) > ERROR_THRESHOLD) {
        fail++;
      }
    }
}

}
}
}
//append writeback of scalar variables
}

