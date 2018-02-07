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
#define PERCENT_DIFF_ERROR_THRESHOLD 0.01


 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_0(float * a,float * c_cpu,float * c_gpu,float * b);
 
void init(float *a, float *b, float *c_cpu, float *c_gpu)
{
  int i, j;
    

	ipmacc_prompt((char*)"IPMACC: memory allocation c_cpu\n");
acc_present_or_create((void*)c_cpu,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation c_gpu\n");
acc_present_or_create((void*)c_gpu,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation b\n");
acc_present_or_create((void*)b,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation a\n");
acc_present_or_create((void*)a,(999999+0)*sizeof(float ));
	ipmacc_prompt((char*)"IPMACC: memory copyin c_cpu\n");
acc_pcopyin((void*)c_cpu,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin c_gpu\n");
acc_pcopyin((void*)c_gpu,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin b\n");
acc_pcopyin((void*)b,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin a\n");
acc_pcopyin((void*)a,(999999+0)*sizeof(float ));


{


    


/* kernel call statement [0, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 0 > gridDim: %d\tblockDim: %d\n",(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_0<<<(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(float *)acc_deviceptr((void*)a),
(float *)acc_deviceptr((void*)c_cpu),
(float *)acc_deviceptr((void*)c_gpu),
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
	ipmacc_prompt((char*)"IPMACC: memory copyout c_cpu\n");
acc_copyout_and_keep((void*)c_cpu,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout c_gpu\n");
acc_copyout_and_keep((void*)c_gpu,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout b\n");
acc_copyout_and_keep((void*)b,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout a\n");
acc_copyout_and_keep((void*)a,(999999+0)*sizeof(float ));



}



 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_1(float * a,float * b,float * c);
 
void sum_GPU(float *a, float *b, float *c)
{
  int i, j;

    #pragma omp target device (GPU_DEVICE)
    #pragma omp target map(to: a[0:SIZE*SIZE], b[0:SIZE*SIZE]) map(tofrom: c[0:SIZE*SIZE])
  {
  #pragma omp parallel for collapse(2)
        

	ipmacc_prompt((char*)"IPMACC: memory allocation a\n");
acc_present_or_create((void*)a,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation b\n");
acc_present_or_create((void*)b,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation c\n");
acc_present_or_create((void*)c,(999999+0)*sizeof(float ));
	ipmacc_prompt((char*)"IPMACC: memory copyin a\n");
acc_pcopyin((void*)a,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin b\n");
acc_pcopyin((void*)b,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin c\n");
acc_pcopyin((void*)c,(999999+0)*sizeof(float ));


{


        


/* kernel call statement [1, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 1 > gridDim: %d\tblockDim: %d\n",(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_1<<<(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(float *)acc_deviceptr((void*)a),
(float *)acc_deviceptr((void*)b),
(float *)acc_deviceptr((void*)c));
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
acc_copyout_and_keep((void*)a,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout b\n");
acc_copyout_and_keep((void*)b,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout c\n");
acc_copyout_and_keep((void*)c,(999999+0)*sizeof(float ));



  }
}



 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_2(float * a,float * b,float * c);
 
void sum_CPU(float *a, float *b, float *c)
{
  int i, j;
    

	ipmacc_prompt((char*)"IPMACC: memory allocation c\n");
acc_present_or_create((void*)c,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation b\n");
acc_present_or_create((void*)b,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation a\n");
acc_present_or_create((void*)a,(999999+0)*sizeof(float ));
	ipmacc_prompt((char*)"IPMACC: memory copyin c\n");
acc_pcopyin((void*)c,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin b\n");
acc_pcopyin((void*)b,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin a\n");
acc_pcopyin((void*)a,(999999+0)*sizeof(float ));


{


    


/* kernel call statement [2, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 2 > gridDim: %d\tblockDim: %d\n",(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_2<<<(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(float *)acc_deviceptr((void*)a),
(float *)acc_deviceptr((void*)b),
(float *)acc_deviceptr((void*)c));
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
	ipmacc_prompt((char*)"IPMACC: memory copyout c\n");
acc_copyout_and_keep((void*)c,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout b\n");
acc_copyout_and_keep((void*)b,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout a\n");
acc_copyout_and_keep((void*)a,(999999+0)*sizeof(float ));



}

 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_3(float * b_gpu,float * b_cpu,int  fail);
 
void compareResults(float *b_cpu, float *b_gpu)
{
  int i, j, fail;
  fail = 0;

    

	ipmacc_prompt((char*)"IPMACC: memory allocation b_cpu\n");
acc_present_or_create((void*)b_cpu,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation b_gpu\n");
acc_present_or_create((void*)b_gpu,(999999+0)*sizeof(float ));
	ipmacc_prompt((char*)"IPMACC: memory copyin b_cpu\n");
acc_pcopyin((void*)b_cpu,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin b_gpu\n");
acc_pcopyin((void*)b_gpu,(999999+0)*sizeof(float ));


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
acc_copyout_and_keep((void*)b_cpu,(999999+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout b_gpu\n");
acc_copyout_and_keep((void*)b_gpu,(999999+0)*sizeof(float ));




  
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

int main(int argc, char *argv[])
{
  double t_start, t_end;
  float *a, *b, *c_cpu, *c_gpu;

  a = (float*)malloc(sizeof(float) * SIZE * SIZE);
  b = (float*)malloc(sizeof(float) * SIZE * SIZE);
  c_cpu = (float*)malloc(sizeof(float) * SIZE * SIZE);
  c_gpu = (float*)malloc(sizeof(float) * SIZE * SIZE);

  fprintf(stdout, "<< Matrix Sum >>\n");

  init(a, b, c_cpu, c_gpu);

  t_start = rtclock();
  sum_GPU(a, b, c_gpu);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  sum_CPU(a, b, c_cpu);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(c_cpu, c_gpu);

  free(a);
  free(b);
  free(c_cpu);
  free(c_gpu);

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

 __global__ void __generated_kernel_region_0(float * a,float * c_cpu,float * c_gpu,float * b){
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
      a [i * SIZE + j] = (float)i + j;
      b [i * SIZE + j] = (float)i + j;
      c_cpu [i * SIZE + j] = 0.0f;
      c_gpu [i * SIZE + j] = 0.0f;
    }
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_1(float * a,float * b,float * c){
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
        c [i * SIZE + j] = a [i * SIZE + j] + b [i * SIZE + j];
      }
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_2(float * a,float * b,float * c){
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
      c [i * SIZE + j] = a [i * SIZE + j] + b [i * SIZE + j];
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

