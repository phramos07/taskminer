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


#define ERROR_THRESHOLD 0.05

#define GPU_DEVICE 1


#define NI 8192
#define NJ 8192


typedef float DATA_TYPE;

void CPU__conv2D(DATA_TYPE* A, DATA_TYPE* B)
{
  int i, j;
  DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

  c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
  c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
  c13 = +0.4;  c23 = +0.7;  c33 = +0.10;

  for (i = 1; i < NI - 1; ++i) { 
    for (j = 1; j < NJ - 1; ++j) { 
      B [i * NJ + j] = c11 * A [(i - 1) * NJ + (j - 1)] + c12 * A [(i + 0) * NJ + (j - 1)] + c13 * A [(i + 1) * NJ + (j - 1)]
                       + c21 * A [(i - 1) * NJ + (j + 0)] + c22 * A [(i + 0) * NJ + (j + 0)] + c23 * A [(i + 1) * NJ + (j + 0)]
                       + c31 * A [(i - 1) * NJ + (j + 1)] + c32 * A [(i + 0) * NJ + (j + 1)] + c33 * A [(i + 1) * NJ + (j + 1)];
    }
  }
}

  __global__ void __generated_kernel_region_0(DATA_TYPE * A,DATA_TYPE * B,DATA_TYPE  c23,DATA_TYPE  c21,DATA_TYPE  c13,DATA_TYPE  c12,DATA_TYPE  c11,DATA_TYPE  c32,DATA_TYPE  c31,DATA_TYPE  c33,DATA_TYPE  c22);
 
void GPU__conv2D(DATA_TYPE* A, DATA_TYPE* B)
{
  int i, j;
  DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

  c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
  c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
  c13 = +0.4;  c23 = +0.7;  c33 = +0.10;

  

	ipmacc_prompt((char*)"IPMACC: memory allocation A\n");
acc_present_or_create((void*)A,(67108863+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation B\n");
acc_present_or_create((void*)B,(67100670+8193)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin A\n");
acc_pcopyin((void*)A,(67108863+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin B\n");
acc_pcopyin((void*)B,(67100670+8193)*sizeof(DATA_TYPE ));


{


  


/* kernel call statement [0, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 0 > gridDim: %d\tblockDim: %d\n",(((abs((int)(((NI-1)))-(1+0)))/(1)))/256+(((((abs((int)(((NI-1)))-(1+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_0<<<(((abs((int)(((NI-1)))-(1+0)))/(1)))/256+(((((abs((int)(((NI-1)))-(1+0)))/(1)))%(256))==0?0:1),256>>>(
(DATA_TYPE *)acc_deviceptr((void*)A),
(DATA_TYPE *)acc_deviceptr((void*)B),
c23,
c21,
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
acc_copyout_and_keep((void*)A,(67108863+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout B\n");
acc_copyout_and_keep((void*)B,(67100670+8193)*sizeof(DATA_TYPE ));



}

void init(DATA_TYPE* A)
{
  int i, j;

  for (i = 0; i < NI; ++i) {
    for (j = 0; j < NJ; ++j) {
      A [i * NJ + j] = (float)rand() / RAND_MAX;
    }
  }
}

void compareResults(DATA_TYPE* B, DATA_TYPE* B_GPU)
{
  int i, j, fail;
  fail = 0;

  
  for (i = 1; i < (NI - 1); i++) {
    for (j = 1; j < (NJ - 1); j++) {
      if (percentDiff(B [i * NJ + j], B_GPU [i * NJ + j]) > ERROR_THRESHOLD) {
        printf("%f %f\n", B [i * NJ + j], B_GPU [i * NJ + j]);
        fail++;
      }
    }
  }

  
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", ERROR_THRESHOLD, fail);
}

int main(int argc, char *argv[])
{
  double t_start, t_end, t_start_OMP, t_end_OMP;

  DATA_TYPE* A;
  DATA_TYPE* B;
  DATA_TYPE* B_OMP;

  A = (DATA_TYPE*)malloc(NI * NJ * sizeof(DATA_TYPE));
  B = (DATA_TYPE*)malloc(NI * NJ * sizeof(DATA_TYPE));
  B_OMP = (DATA_TYPE*)malloc(NI * NJ * sizeof(DATA_TYPE));

  fprintf(stdout, ">> Two dimensional (2D) convolution <<\n");

  
  init(A);

  t_start_OMP = rtclock();
  GPU__conv2D(A, B_OMP);
  t_end_OMP = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end_OMP - t_start_OMP); 

  t_start = rtclock();
  CPU__conv2D(A, B);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start); 

  compareResults(B, B_OMP);

  free(A);
  free(B);
  free(B_OMP);

  return 0;
}



 __global__ void __generated_kernel_region_0(DATA_TYPE * A,DATA_TYPE * B,DATA_TYPE  c23,DATA_TYPE  c21,DATA_TYPE  c13,DATA_TYPE  c12,DATA_TYPE  c11,DATA_TYPE  c32,DATA_TYPE  c31,DATA_TYPE  c33,DATA_TYPE  c22){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  j;
{
{
{
 i=1+(__kernel_getuid_x);
if( i < NI - 1)
{
for(j = 1; j < NJ - 1; ++j)
{
      B [i * NJ + j] = c11 * A [(i - 1) * NJ + (j - 1)] + c12 * A [(i + 0) * NJ + (j - 1)] + c13 * A [(i + 1) * NJ + (j - 1)]
                       + c21 * A [(i - 1) * NJ + (j + 0)] + c22 * A [(i + 0) * NJ + (j + 0)] + c23 * A [(i + 1) * NJ + (j + 0)]
                       + c31 * A [(i - 1) * NJ + (j + 1)] + c32 * A [(i + 0) * NJ + (j + 1)] + c33 * A [(i + 1) * NJ + (j + 1)];
    }
}

}
}
}
//append writeback of scalar variables
}

