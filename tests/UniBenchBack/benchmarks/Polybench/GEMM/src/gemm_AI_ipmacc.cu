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


#define PERCENT_DIFF_ERROR_THRESHOLD 0.05


#define NI 512
#define NJ 512
#define NK 512


#define ALPHA 32412.0f
#define BETA 2123.0f


typedef float DATA_TYPE;

#define GPU_DEVICE 1

void CPU__gemm(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
  int i, j, k;

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      C [i * NJ + j] *= BETA;

      for (k = 0; k < NK; ++k) {
        C [i * NJ + j] += ALPHA * A [i * NK + k] * B [k * NJ + j];
      }
    }
  }
}

  __global__ void __generated_kernel_region_0(DATA_TYPE * A,DATA_TYPE * B,DATA_TYPE * C);
 
void GPU__gemm(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
  int i, j, k;

  

	ipmacc_prompt((char*)"IPMACC: memory allocation A\n");
acc_present_or_create((void*)A,(262143+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation B\n");
acc_present_or_create((void*)B,(262143+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation C\n");
acc_present_or_create((void*)C,(262143+0)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin A\n");
acc_pcopyin((void*)A,(262143+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin B\n");
acc_pcopyin((void*)B,(262143+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin C\n");
acc_pcopyin((void*)C,(262143+0)*sizeof(DATA_TYPE ));


{


  


/* kernel call statement [0, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 0 > gridDim: %d\tblockDim: %d\n",(((abs((int)((NI))-(0+0)))/(1)))/256+(((((abs((int)((NI))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_0<<<(((abs((int)((NI))-(0+0)))/(1)))/256+(((((abs((int)((NI))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(DATA_TYPE *)acc_deviceptr((void*)A),
(DATA_TYPE *)acc_deviceptr((void*)B),
(DATA_TYPE *)acc_deviceptr((void*)C));
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
acc_copyout_and_keep((void*)A,(262143+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout B\n");
acc_copyout_and_keep((void*)B,(262143+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout C\n");
acc_copyout_and_keep((void*)C,(262143+0)*sizeof(DATA_TYPE ));



}

void init(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *C_OMP)
{
  int i, j;

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NK; j++) {
      A [i * NK + j] = ((DATA_TYPE)i * j) / NI;
    }
  }

  for (i = 0; i < NK; i++) {
    for (j = 0; j < NJ; j++) {
      B [i * NJ + j] = ((DATA_TYPE)i * j + 1) / NJ;
    }
  }

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      C [i * NJ + j] = ((DATA_TYPE)i * j + 2) / NJ;
      C_OMP [i * NJ + j] = ((DATA_TYPE)i * j + 2) / NJ;
    }
  }
}

void compareResults(DATA_TYPE* C, DATA_TYPE* C_outputFromGpu)
{
  int i, j, fail;
  fail = 0;

  
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      if (percentDiff(C [i * NJ + j], C_outputFromGpu [i * NJ + j]) > PERCENT_DIFF_ERROR_THRESHOLD) {
        fail++;
      }
    }
  }

  
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

int main(int argc, char *argv[])
{
  double t_start, t_end;

  DATA_TYPE* A;
  DATA_TYPE* B;
  DATA_TYPE* C;
  DATA_TYPE* C_outputFromGpu;

  A = (DATA_TYPE*)malloc(NI * NK * sizeof(DATA_TYPE));
  B = (DATA_TYPE*)malloc(NK * NJ * sizeof(DATA_TYPE));
  C = (DATA_TYPE*)malloc(NI * NJ * sizeof(DATA_TYPE));
  C_outputFromGpu = (DATA_TYPE*)malloc(NI * NJ * sizeof(DATA_TYPE));

  fprintf(stdout, "<< Matrix-multiply C=alpha.A.B+beta.C >>\n");

  init(A, B, C, C_outputFromGpu);

  t_start = rtclock();
  GPU__gemm(A, B, C_outputFromGpu);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  CPU__gemm(A, B, C);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(C, C_outputFromGpu);

  free(A);
  free(B);
  free(C);
  free(C_outputFromGpu);

  return 0;
}



 __global__ void __generated_kernel_region_0(DATA_TYPE * A,DATA_TYPE * B,DATA_TYPE * C){
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
if( i < NI)
{
for(j = 0; j < NJ; j++)
{
      C [i * NJ + j] *= BETA;
for(k = 0; k < NK; ++k)
{
        C [i * NJ + j] += ALPHA * A [i * NK + k] * B [k * NJ + j];
      }
}
}

}
}
}
//append writeback of scalar variables
}

