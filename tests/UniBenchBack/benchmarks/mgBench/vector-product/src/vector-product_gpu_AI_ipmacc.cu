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
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05


 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_0(float * A,float * B);
 
void init_arrays(float *A, float *B)
{
  int i;
    

	ipmacc_prompt((char*)"IPMACC: memory allocation B\n");
acc_present_or_create((void*)B,(999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation A\n");
acc_present_or_create((void*)A,(999+0)*sizeof(float ));
	ipmacc_prompt((char*)"IPMACC: memory copyin B\n");
acc_pcopyin((void*)B,(999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin A\n");
acc_pcopyin((void*)A,(999+0)*sizeof(float ));


{


    


/* kernel call statement [0, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 0 > gridDim: %d\tblockDim: %d\n",(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_0<<<(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
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
	ipmacc_prompt((char*)"IPMACC: memory copyout B\n");
acc_copyout_and_keep((void*)B,(999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout A\n");
acc_copyout_and_keep((void*)A,(999+0)*sizeof(float ));



}

 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_1(float * A,float * B,float * C);
 
void product_GPU(float *A, float *B, float *C)
{
  int i;

    #pragma omp target device (GPU_DEVICE)
    #pragma omp target map(to: A[0:SIZE], B[0:SIZE]) map(from: C[0:SIZE])
  {
  #pragma omp parallel for
        

	ipmacc_prompt((char*)"IPMACC: memory allocation C\n");
acc_present_or_create((void*)C,(999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation B\n");
acc_present_or_create((void*)B,(999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation A\n");
acc_present_or_create((void*)A,(999+0)*sizeof(float ));
	ipmacc_prompt((char*)"IPMACC: memory copyin C\n");
acc_pcopyin((void*)C,(999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin B\n");
acc_pcopyin((void*)B,(999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin A\n");
acc_pcopyin((void*)A,(999+0)*sizeof(float ));


{


        


/* kernel call statement [1, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 1 > gridDim: %d\tblockDim: %d\n",(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_1<<<(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(float *)acc_deviceptr((void*)A),
(float *)acc_deviceptr((void*)B),
(float *)acc_deviceptr((void*)C));
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
	ipmacc_prompt((char*)"IPMACC: memory copyout C\n");
acc_copyout_and_keep((void*)C,(999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout B\n");
acc_copyout_and_keep((void*)B,(999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout A\n");
acc_copyout_and_keep((void*)A,(999+0)*sizeof(float ));



  }
}

 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_2(float * A,float * B,float * C);
 
void product_CPU(float *A, float *B, float *C)
{
  int i;

    

	ipmacc_prompt((char*)"IPMACC: memory allocation B\n");
acc_present_or_create((void*)B,(999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation C\n");
acc_present_or_create((void*)C,(999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation A\n");
acc_present_or_create((void*)A,(999+0)*sizeof(float ));
	ipmacc_prompt((char*)"IPMACC: memory copyin B\n");
acc_pcopyin((void*)B,(999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin C\n");
acc_pcopyin((void*)C,(999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin A\n");
acc_pcopyin((void*)A,(999+0)*sizeof(float ));


{


    


/* kernel call statement [2, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 2 > gridDim: %d\tblockDim: %d\n",(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_2<<<(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(float *)acc_deviceptr((void*)A),
(float *)acc_deviceptr((void*)B),
(float *)acc_deviceptr((void*)C));
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
	ipmacc_prompt((char*)"IPMACC: memory copyout B\n");
acc_copyout_and_keep((void*)B,(999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout C\n");
acc_copyout_and_keep((void*)C,(999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout A\n");
acc_copyout_and_keep((void*)A,(999+0)*sizeof(float ));



}

 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_3(float * A,float * A_outputFromGpu,int  fail);
 
void compareResults(float *A, float *A_outputFromGpu)
{
  int i, j, fail;
  fail = 0;

    

	ipmacc_prompt((char*)"IPMACC: memory allocation A_outputFromGpu\n");
acc_present_or_create((void*)A_outputFromGpu,(999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation A\n");
acc_present_or_create((void*)A,(999+0)*sizeof(float ));
	ipmacc_prompt((char*)"IPMACC: memory copyin A_outputFromGpu\n");
acc_pcopyin((void*)A_outputFromGpu,(999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin A\n");
acc_pcopyin((void*)A,(999+0)*sizeof(float ));


{


    


/* kernel call statement [3, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 3 > gridDim: %d\tblockDim: %d\n",(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_3<<<(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(float *)acc_deviceptr((void*)A),
(float *)acc_deviceptr((void*)A_outputFromGpu),
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
	ipmacc_prompt((char*)"IPMACC: memory copyout A_outputFromGpu\n");
acc_copyout_and_keep((void*)A_outputFromGpu,(999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout A\n");
acc_copyout_and_keep((void*)A,(999+0)*sizeof(float ));




  
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

int main(int argc, char *argv[])
{
  double t_start, t_end;
  float *A, *B, *C_CPU, *C_GPU;

  A = (float*)malloc(sizeof(float) * SIZE);
  B = (float*)malloc(sizeof(float) * SIZE);
  C_CPU = (float*)malloc(sizeof(float) * SIZE);
  C_GPU = (float*)malloc(sizeof(float) * SIZE);


  fprintf(stdout, "<< Vector Product >>\n");
  init_arrays(A, B);

  t_start = rtclock();
  product_GPU(A, B, C_GPU);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);


  t_start = rtclock();
  product_CPU(A, B, C_CPU);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(C_CPU, C_GPU);

  free(A);
  free(B);
  free(C_CPU);
  free(C_GPU);

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

 __global__ void __generated_kernel_region_0(float * A,float * B){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < SIZE)
{
    A [i] = (float)i + 3 * i;
    B [i] = (float)i + 2 * i;
  }

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_1(float * A,float * B,float * C){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < SIZE)
{
      C [i] = A [i] * B [i];
    }

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_2(float * A,float * B,float * C){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < SIZE)
{
    C [i] = A [i] * B [i];
  }

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_3(float * A,float * A_outputFromGpu,int  fail){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < SIZE)
{
    if (__accelerator_percentDiff(A [i], A_outputFromGpu [i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
      fail++;
      
    }
  }

}
}
}
//append writeback of scalar variables
}

