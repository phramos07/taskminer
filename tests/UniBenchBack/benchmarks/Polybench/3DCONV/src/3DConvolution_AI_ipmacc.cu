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



#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>


#include "../../common/polybenchUtilFuncts.h"


#define ERROR_THRESHOLD 0.5

#define GPU_DEVICE 1


#define NI 512
#define NJ 512
#define NK 512


typedef float DATA_TYPE;

 __device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_0(DATA_TYPE * A,DATA_TYPE  c21,DATA_TYPE * B,DATA_TYPE  c23,DATA_TYPE  c13,DATA_TYPE  c12,DATA_TYPE  c11,DATA_TYPE  c32,DATA_TYPE  c31,DATA_TYPE  c33,DATA_TYPE  c22);
 
void CPU__conv3D(DATA_TYPE* A, DATA_TYPE* B)
{
  int i, j, k;
  DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

  c11 = +2;  c21 = +5;  c31 = -8;
  c12 = -3;  c22 = +6;  c32 = -9;
  c13 = +4;  c23 = +7;  c33 = +10;

  

	ipmacc_prompt((char*)"IPMACC: memory allocation B\n");
acc_present_or_create((void*)B,(133955070+262657)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation A\n");
acc_present_or_create((void*)A,(134217727+0)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin B\n");
acc_pcopyin((void*)B,(133955070+262657)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin A\n");
acc_pcopyin((void*)A,(134217727+0)*sizeof(DATA_TYPE ));


{


  


/* kernel call statement [0, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 0 > gridDim: %d\tblockDim: %d\n",(((abs((int)(((NJ-1)))-(1+0)))/(1)))/256+(((((abs((int)(((NJ-1)))-(1+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_0<<<(((abs((int)(((NJ-1)))-(1+0)))/(1)))/256+(((((abs((int)(((NJ-1)))-(1+0)))/(1)))%(256))==0?0:1),256>>>(
(DATA_TYPE *)acc_deviceptr((void*)A),
c21,
(DATA_TYPE *)acc_deviceptr((void*)B),
c23,
c13,
c12,
c11,
c32,
c31,
c33,
c22);
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
acc_copyout_and_keep((void*)B,(133955070+262657)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout A\n");
acc_copyout_and_keep((void*)A,(134217727+0)*sizeof(DATA_TYPE ));



}

 __device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_1(DATA_TYPE * A,DATA_TYPE  c21,DATA_TYPE * B,DATA_TYPE  c23,DATA_TYPE  c13,DATA_TYPE  c12,DATA_TYPE  c11,DATA_TYPE  c32,DATA_TYPE  c31,DATA_TYPE  c33,DATA_TYPE  c22);
 
void GPU__conv3D(DATA_TYPE* A, DATA_TYPE* B)
{
  int i, j, k;
  DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

  c11 = +2;  c21 = +5;  c31 = -8;
  c12 = -3;  c22 = +6;  c32 = -9;
  c13 = +4;  c23 = +7;  c33 = +10;

  

	ipmacc_prompt((char*)"IPMACC: memory allocation A\n");
acc_present_or_create((void*)A,(134217727+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation B\n");
acc_present_or_create((void*)B,(133955070+262657)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin A\n");
acc_pcopyin((void*)A,(134217727+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin B\n");
acc_pcopyin((void*)B,(133955070+262657)*sizeof(DATA_TYPE ));


{


  


/* kernel call statement [1, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 1 > gridDim: %d\tblockDim: %d\n",(((abs((int)(((NJ-1)))-(1+0)))/(1)))/256+(((((abs((int)(((NJ-1)))-(1+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_1<<<(((abs((int)(((NJ-1)))-(1+0)))/(1)))/256+(((((abs((int)(((NJ-1)))-(1+0)))/(1)))%(256))==0?0:1),256>>>(
(DATA_TYPE *)acc_deviceptr((void*)A),
c21,
(DATA_TYPE *)acc_deviceptr((void*)B),
c23,
c13,
c12,
c11,
c32,
c31,
c33,
c22);
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
acc_copyout_and_keep((void*)A,(134217727+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout B\n");
acc_copyout_and_keep((void*)B,(133955070+262657)*sizeof(DATA_TYPE ));



}

void init(DATA_TYPE* A)
{
  int i, j, k;

  for (i = 0; i < NI; ++i) {
    for (j = 0; j < NJ; ++j) {
      for (k = 0; k < NK; ++k) {
        A [i * (NK * NJ) + j * NK + k] = i % 12 + 2 * (j % 7) + 3 * (k % 13);
      }
    }
  }
}

 __device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_2(DATA_TYPE * B_GPU,DATA_TYPE * B,int  fail);
 
void compareResults(DATA_TYPE* B, DATA_TYPE* B_GPU)
{
  int i, j, k, fail;
  fail = 0;

  
  

	ipmacc_prompt((char*)"IPMACC: memory allocation B_GPU\n");
acc_present_or_create((void*)B_GPU,(133955070+262657)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation B\n");
acc_present_or_create((void*)B,(133955070+262657)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin B_GPU\n");
acc_pcopyin((void*)B_GPU,(133955070+262657)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin B\n");
acc_pcopyin((void*)B,(133955070+262657)*sizeof(DATA_TYPE ));


{


  


/* kernel call statement [2, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 2 > gridDim: %d\tblockDim: %d\n",(((abs((int)(((NI-1)))-(1+0)))/(1)))/256+(((((abs((int)(((NI-1)))-(1+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_2<<<(((abs((int)(((NI-1)))-(1+0)))/(1)))/256+(((((abs((int)(((NI-1)))-(1+0)))/(1)))%(256))==0?0:1),256>>>(
(DATA_TYPE *)acc_deviceptr((void*)B_GPU),
(DATA_TYPE *)acc_deviceptr((void*)B),
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
	ipmacc_prompt((char*)"IPMACC: memory copyout B_GPU\n");
acc_copyout_and_keep((void*)B_GPU,(133955070+262657)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout B\n");
acc_copyout_and_keep((void*)B,(133955070+262657)*sizeof(DATA_TYPE ));




  
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", ERROR_THRESHOLD, fail);
}

int main(int argc, char *argv[])
{
  double t_start, t_end;

  DATA_TYPE* A;
  DATA_TYPE* B;
  DATA_TYPE* B_GPU;

  A = (DATA_TYPE*)malloc(NI * NJ * NK * sizeof(DATA_TYPE));
  B = (DATA_TYPE*)malloc(NI * NJ * NK * sizeof(DATA_TYPE));
  B_GPU = (DATA_TYPE*)malloc(NI * NJ * NK * sizeof(DATA_TYPE));

  fprintf(stdout, ">> Three dimensional (3D) convolution <<\n");

  init(A);

  t_start = rtclock();
  GPU__conv3D(A, B_GPU);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  CPU__conv3D(A, B);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(B, B_GPU);

  free(A);
  free(B);
  free(B_GPU);

  return 0;
}


__device__ float __accelerator_percentDiff( double val1 , double val2 ) {
  return  ( val1  - val2 ) ; 
}

 __global__ void __generated_kernel_region_0(DATA_TYPE * A,DATA_TYPE  c21,DATA_TYPE * B,DATA_TYPE  c23,DATA_TYPE  c13,DATA_TYPE  c12,DATA_TYPE  c11,DATA_TYPE  c32,DATA_TYPE  c31,DATA_TYPE  c33,DATA_TYPE  c22){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  k;
int  j;
{
{
{
 j=1+(__kernel_getuid_x);
if( j < NJ - 1)
{
for(i = 1; i < NI - 1; ++i)
{
for(k = 1; k < NK - 1; ++k)
{
        B [i * (NK * NJ) + j * NK + k] = c11 * A [(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c13 * A [(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)]
                                         + c21 * A [(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c23 * A [(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)]
                                         + c31 * A [(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c33 * A [(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)]
                                         + c12 * A [(i + 0) * (NK * NJ) + (j - 1) * NK + (k + 0)] + c22 * A [(i + 0) * (NK * NJ) + (j + 0) * NK + (k + 0)]
                                         + c32 * A [(i + 0) * (NK * NJ) + (j + 1) * NK + (k + 0)] + c11 * A [(i - 1) * (NK * NJ) + (j - 1) * NK + (k + 1)]
                                         + c13 * A [(i + 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] + c21 * A [(i - 1) * (NK * NJ) + (j + 0) * NK + (k + 1)]
                                         + c23 * A [(i + 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] + c31 * A [(i - 1) * (NK * NJ) + (j + 1) * NK + (k + 1)]
                                         + c33 * A [(i + 1) * (NK * NJ) + (j + 1) * NK + (k + 1)];
      }
}
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_1(DATA_TYPE * A,DATA_TYPE  c21,DATA_TYPE * B,DATA_TYPE  c23,DATA_TYPE  c13,DATA_TYPE  c12,DATA_TYPE  c11,DATA_TYPE  c32,DATA_TYPE  c31,DATA_TYPE  c33,DATA_TYPE  c22){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  k;
int  j;
{
{
{
 j=1+(__kernel_getuid_x);
if( j < NJ - 1)
{
for(i = 1; i < NI - 1; ++i)
{
      int k;
for(k = 1; k < NK - 1; ++k)
{
        B [i * (NK * NJ) + j * NK + k] = c11 * A [(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c13 * A [(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)]
                                         + c21 * A [(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c23 * A [(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)]
                                         + c31 * A [(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c33 * A [(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)]
                                         + c12 * A [(i + 0) * (NK * NJ) + (j - 1) * NK + (k + 0)] + c22 * A [(i + 0) * (NK * NJ) + (j + 0) * NK + (k + 0)]
                                         + c32 * A [(i + 0) * (NK * NJ) + (j + 1) * NK + (k + 0)] + c11 * A [(i - 1) * (NK * NJ) + (j - 1) * NK + (k + 1)]
                                         + c13 * A [(i + 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] + c21 * A [(i - 1) * (NK * NJ) + (j + 0) * NK + (k + 1)]
                                         + c23 * A [(i + 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] + c31 * A [(i - 1) * (NK * NJ) + (j + 1) * NK + (k + 1)]
                                         + c33 * A [(i + 1) * (NK * NJ) + (j + 1) * NK + (k + 1)];
      }
}
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_2(DATA_TYPE * B_GPU,DATA_TYPE * B,int  fail){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  k;
int  j;
{
{
{
 i=1+(__kernel_getuid_x);
if( i < NI - 1)
{
for(j = 1; j < NJ - 1; ++j)
{
for(k = 1; k < NK - 1; ++k)
{
        if (__accelerator_percentDiff(B [i * (NK * NJ) + j * NK + k], B_GPU [i * (NK * NJ) + j * NK + k]) > ERROR_THRESHOLD) {
          fail++;
        }
      }
}
}

}
}
}
//append writeback of scalar variables
}

