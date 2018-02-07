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


#include "../../common/polybenchUtilFuncts.h"


#define PERCENT_DIFF_ERROR_THRESHOLD 0.5


#define NX 8192
#define NY 8192

#define GPU_DEVICE 1

#ifndef M_PI
#define M_PI 3.14159
#endif


typedef float DATA_TYPE;

void init_array(DATA_TYPE *x, DATA_TYPE *A)
{
  int i, j;

  for (i = 0; i < NX; i++) {
    x [i] = i * M_PI;
    for (j = 0; j < NY; j++) {
      A [i * NY + j] = ((DATA_TYPE)i * (j)) / NX;
    }
  }
}

void compareResults(DATA_TYPE *z, DATA_TYPE *z_outputFromGpu)
{
  int i, fail;
  fail = 0;

  for (i = 0; i < NY; i++) {
    if (percentDiff(z [i], z_outputFromGpu [i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
      fail++;
    }
  }

  
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void CPU__atax(DATA_TYPE* A, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp)
{
  int i, j;

  for (i = 0; i < NY; i++) {
    y [i] = 0;
  }

  for (i = 0; i < NX; i++) {
    tmp [i] = 0;

    for (j = 0; j < NY; j++) {
      tmp [i] = tmp [i] + A [i * NY + j] * x [j];
    }

    for (j = 0; j < NY; j++) {
      y [j] = y [j] + A [i * NY + j] * tmp [i];
    }
  }
}

  __global__ void __generated_kernel_region_0(DATA_TYPE * y);
 
  __global__ void __generated_kernel_region_1(DATA_TYPE * tmp,DATA_TYPE * A,DATA_TYPE * x);
 
  __global__ void __generated_kernel_region_2(DATA_TYPE * A,DATA_TYPE * tmp,DATA_TYPE * y);
 
void GPU__atax(DATA_TYPE* A, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp)
{
  int i, j;

  

	ipmacc_prompt((char*)"IPMACC: memory allocation y\n");
acc_present_or_create((void*)y,(8191+0)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin y\n");
acc_pcopyin((void*)y,(8191+0)*sizeof(DATA_TYPE ));


{


  


/* kernel call statement [0, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 0 > gridDim: %d\tblockDim: %d\n",(((abs((int)((NY))-(0+0)))/(1)))/256+(((((abs((int)((NY))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_0<<<(((abs((int)((NY))-(0+0)))/(1)))/256+(((((abs((int)((NY))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(DATA_TYPE *)acc_deviceptr((void*)y));
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
	ipmacc_prompt((char*)"IPMACC: memory copyout y\n");
acc_copyout_and_keep((void*)y,(8191+0)*sizeof(DATA_TYPE ));




  

	ipmacc_prompt((char*)"IPMACC: memory allocation A\n");
acc_present_or_create((void*)A,(67108863+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation x\n");
acc_present_or_create((void*)x,(8191+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation tmp\n");
acc_present_or_create((void*)tmp,(8191+0)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin A\n");
acc_pcopyin((void*)A,(67108863+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin x\n");
acc_pcopyin((void*)x,(8191+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin tmp\n");
acc_pcopyin((void*)tmp,(8191+0)*sizeof(DATA_TYPE ));


{


  


/* kernel call statement [1, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 1 > gridDim: %d\tblockDim: %d\n",(((abs((int)((NX))-(0+0)))/(1)))/256+(((((abs((int)((NX))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_1<<<(((abs((int)((NX))-(0+0)))/(1)))/256+(((((abs((int)((NX))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(DATA_TYPE *)acc_deviceptr((void*)tmp),
(DATA_TYPE *)acc_deviceptr((void*)A),
(DATA_TYPE *)acc_deviceptr((void*)x));
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
acc_copyout_and_keep((void*)A,(67108863+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout x\n");
acc_copyout_and_keep((void*)x,(8191+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout tmp\n");
acc_copyout_and_keep((void*)tmp,(8191+0)*sizeof(DATA_TYPE ));




  
  

	ipmacc_prompt((char*)"IPMACC: memory allocation A\n");
acc_present_or_create((void*)A,(67108863+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation y\n");
acc_present_or_create((void*)y,(8191+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation tmp\n");
acc_present_or_create((void*)tmp,(8191+0)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin A\n");
acc_pcopyin((void*)A,(67108863+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin y\n");
acc_pcopyin((void*)y,(8191+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin tmp\n");
acc_pcopyin((void*)tmp,(8191+0)*sizeof(DATA_TYPE ));


{


  


/* kernel call statement [2, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 2 > gridDim: %d\tblockDim: %d\n",(((abs((int)((NY))-(0+0)))/(1)))/256+(((((abs((int)((NY))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_2<<<(((abs((int)((NY))-(0+0)))/(1)))/256+(((((abs((int)((NY))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(DATA_TYPE *)acc_deviceptr((void*)A),
(DATA_TYPE *)acc_deviceptr((void*)tmp),
(DATA_TYPE *)acc_deviceptr((void*)y));
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
acc_copyout_and_keep((void*)A,(67108863+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout y\n");
acc_copyout_and_keep((void*)y,(8191+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout tmp\n");
acc_copyout_and_keep((void*)tmp,(8191+0)*sizeof(DATA_TYPE ));



}

int main(int argc, char** argv)
{
  double t_start, t_end;

  DATA_TYPE* A;
  DATA_TYPE* x;
  DATA_TYPE* y;
  DATA_TYPE* y_outputFromGpu;
  DATA_TYPE* tmp;

  A = (DATA_TYPE*)malloc(NX * NY * sizeof(DATA_TYPE));
  x = (DATA_TYPE*)malloc(NY * sizeof(DATA_TYPE));
  y = (DATA_TYPE*)malloc(NY * sizeof(DATA_TYPE));
  y_outputFromGpu = (DATA_TYPE*)malloc(NY * sizeof(DATA_TYPE));
  tmp = (DATA_TYPE*)malloc(NX * sizeof(DATA_TYPE));

  fprintf(stdout, "<< Matrix Transpose and Vector Multiplication >>\n");

  init_array(x, A);

  t_start = rtclock();
  GPU__atax(A, x, y_outputFromGpu, tmp);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  CPU__atax(A, x, y, tmp);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(y, y_outputFromGpu);

  free(A);
  free(x);
  free(y);
  free(y_outputFromGpu);
  free(tmp);

  return 0;
}



 __global__ void __generated_kernel_region_0(DATA_TYPE * y){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < NY)
{
    y [i] = 0;
  }

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_1(DATA_TYPE * tmp,DATA_TYPE * A,DATA_TYPE * x){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  j;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < NX)
{
    tmp [i] = 0;
    int j;
for(j = 0; j < NY; j++)
{
      tmp [i] = tmp [i] + A [i * NY + j] * x [j];
    }
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_2(DATA_TYPE * A,DATA_TYPE * tmp,DATA_TYPE * y){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  j;
{
{
{
 j=0+(__kernel_getuid_x);
if( j < NY)
{
for(i = 0; i < NX; i++)
{
      {
        y [j] = y [j] + A [i * NY + j] * tmp [i];
      }
    }
}

}
}
}
//append writeback of scalar variables
}

