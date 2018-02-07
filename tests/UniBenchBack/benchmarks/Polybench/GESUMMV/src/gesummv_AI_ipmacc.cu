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

#define GPU_DEVICE 1


#define N 8192


#define ALPHA 43532.0f
#define BETA 12313.0f


typedef float DATA_TYPE;

void CPU__gesummv(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp)
{
  int i, j;

  for (i = 0; i < N; i++) {
    tmp [i] = 0;
    y [i] = 0;
    for (j = 0; j < N; j++) {
      tmp [i] = A [i * N + j] * x [j] + tmp [i];
      y [i] = B [i * N + j] * x [j] + y [i];
    }

    y [i] = ALPHA * tmp [i] + BETA * y [i];
  }
}

  __global__ void __generated_kernel_region_0(DATA_TYPE * tmp,DATA_TYPE * A,DATA_TYPE * B,DATA_TYPE * y,DATA_TYPE * x);
 
void GPU__gesummv(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp)
{
  int i, j;

  

	ipmacc_prompt((char*)"IPMACC: memory allocation A\n");
acc_present_or_create((void*)A,(67108863+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation B\n");
acc_present_or_create((void*)B,(67108863+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation x\n");
acc_present_or_create((void*)x,(8191+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation y\n");
acc_present_or_create((void*)y,(8191+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation tmp\n");
acc_present_or_create((void*)tmp,(8191+0)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin A\n");
acc_pcopyin((void*)A,(67108863+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin B\n");
acc_pcopyin((void*)B,(67108863+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin x\n");
acc_pcopyin((void*)x,(8191+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin y\n");
acc_pcopyin((void*)y,(8191+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin tmp\n");
acc_pcopyin((void*)tmp,(8191+0)*sizeof(DATA_TYPE ));


{


  


/* kernel call statement [0, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 0 > gridDim: %d\tblockDim: %d\n",(((abs((int)((N))-(0+0)))/(1)))/256+(((((abs((int)((N))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_0<<<(((abs((int)((N))-(0+0)))/(1)))/256+(((((abs((int)((N))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(DATA_TYPE *)acc_deviceptr((void*)tmp),
(DATA_TYPE *)acc_deviceptr((void*)A),
(DATA_TYPE *)acc_deviceptr((void*)B),
(DATA_TYPE *)acc_deviceptr((void*)y),
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
ipmacc_prompt((char*)"IPMACC: memory copyout B\n");
acc_copyout_and_keep((void*)B,(67108863+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout x\n");
acc_copyout_and_keep((void*)x,(8191+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout y\n");
acc_copyout_and_keep((void*)y,(8191+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout tmp\n");
acc_copyout_and_keep((void*)tmp,(8191+0)*sizeof(DATA_TYPE ));



}

void init(DATA_TYPE* A, DATA_TYPE* x)
{
  int i, j;

  for (i = 0; i < N; i++) {
    x [i] = ((DATA_TYPE)i) / N;

    for (j = 0; j < N; j++) {
      A [i * N + j] = ((DATA_TYPE)i * j) / N;
    }
  }
}

void compareResults(DATA_TYPE* y, DATA_TYPE* y_outputFromGpu)
{
  int i, fail;
  fail = 0;

  for (i = 0; i < (N); i++) {
    if (percentDiff(y [i], y_outputFromGpu [i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
      fail++;
    }
  }

  
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

int main(int argc, char *argv[])
{
  double t_start, t_end;

  DATA_TYPE* A;
  DATA_TYPE* B;
  DATA_TYPE* x;
  DATA_TYPE* y;
  DATA_TYPE* y_outputFromGpu;
  DATA_TYPE* tmp;

  A = (DATA_TYPE*)malloc(N * N * sizeof(DATA_TYPE));
  B = (DATA_TYPE*)malloc(N * N * sizeof(DATA_TYPE));
  x = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
  y = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
  y_outputFromGpu = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
  tmp = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));

  fprintf(stdout, "<< Scalar, Vector and Matrix Multiplication >>\n");

  init(A, x);

  t_start = rtclock();
  GPU__gesummv(A, B, x, y_outputFromGpu, tmp);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  CPU__gesummv(A, B, x, y, tmp);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(y, y_outputFromGpu);

  free(A);
  free(B);
  free(x);
  free(y);
  free(y_outputFromGpu);
  free(tmp);

  return 0;
}



 __global__ void __generated_kernel_region_0(DATA_TYPE * tmp,DATA_TYPE * A,DATA_TYPE * B,DATA_TYPE * y,DATA_TYPE * x){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  j;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < N)
{
    tmp [i] = 0;
    y [i] = 0;
for(j = 0; j < N; j++)
{
      tmp [i] = A [i * N + j] * x [j] + tmp [i];
      y [i] = B [i * N + j] * x [j] + y [i];
    }
y [i] = ALPHA * tmp [i] + BETA * y [i];
  }

}
}
}
//append writeback of scalar variables
}

