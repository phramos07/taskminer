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


#define N 8192


typedef float DATA_TYPE;

void init(DATA_TYPE * A, DATA_TYPE * B)
{
  int i;

  for (i = 0; i < N; i++) {
    A [i] = i / 2.0;
    B [i] = ((N - 1) - i) / 3.0;
  }

  return;
}

  __global__ void __generated_kernel_region_0(DATA_TYPE * A,DATA_TYPE * B,DATA_TYPE * C);
 
void GPU__vec_mult(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C)
{
  int i;

  

	ipmacc_prompt((char*)"IPMACC: memory allocation C\n");
acc_present_or_create((void*)C,(8191+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation B\n");
acc_present_or_create((void*)B,(8191+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation A\n");
acc_present_or_create((void*)A,(8191+0)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin C\n");
acc_pcopyin((void*)C,(8191+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin B\n");
acc_pcopyin((void*)B,(8191+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin A\n");
acc_pcopyin((void*)A,(8191+0)*sizeof(DATA_TYPE ));


{


  


/* kernel call statement [0, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 0 > gridDim: %d\tblockDim: %d\n",(((abs((int)((N))-(0+0)))/(1)))/256+(((((abs((int)((N))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_0<<<(((abs((int)((N))-(0+0)))/(1)))/256+(((((abs((int)((N))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
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
	ipmacc_prompt((char*)"IPMACC: memory copyout C\n");
acc_copyout_and_keep((void*)C,(8191+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout B\n");
acc_copyout_and_keep((void*)B,(8191+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout A\n");
acc_copyout_and_keep((void*)A,(8191+0)*sizeof(DATA_TYPE ));



}

void CPU__vec_mult(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C)
{
  int i;
  for (i = 0; i < N; i++) {
    C [i] = A [i] * B [i];
  }
}

void compareResults(DATA_TYPE* B, DATA_TYPE* B_GPU)
{
  int i, fail;
  fail = 0;

  
  for (i = 0; i < N; i++) {
    if (percentDiff(B [i], B_GPU [i]) > ERROR_THRESHOLD) {
      fail++;
    }
  }

  
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", ERROR_THRESHOLD, fail);
}

int main(int argc, char *argv[])
{
  double t_start, t_end, t_start_OMP, t_end_OMP;

  DATA_TYPE* A;
  DATA_TYPE* B;
  DATA_TYPE* C;
  DATA_TYPE* C_OMP;

  A = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
  B = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
  C = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
  C_OMP = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));

  fprintf(stdout, ">> Two vector multiplication <<\n");

  
  init(A, B);

  t_start_OMP = rtclock();
  GPU__vec_mult(A, B, C_OMP);
  t_end_OMP = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end_OMP - t_start_OMP); 

  t_start = rtclock();
  CPU__vec_mult(A, B, C);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start); 

  compareResults(C, C_OMP);

  free(A);
  free(B);
  free(C);
  free(C_OMP);

  return 0;
}



 __global__ void __generated_kernel_region_0(DATA_TYPE * A,DATA_TYPE * B,DATA_TYPE * C){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < N)
{
    C [i] = A [i] * B [i];
  }

}
}
}
//append writeback of scalar variables
}

