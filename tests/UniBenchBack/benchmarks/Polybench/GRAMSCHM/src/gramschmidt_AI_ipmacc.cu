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
#include <math.h>


#include "../../common/polybenchUtilFuncts.h"


#define PERCENT_DIFF_ERROR_THRESHOLD 0.05


#define M 512
#define N 512


typedef float DATA_TYPE;

#define GPU_DEVICE 1

void CPU__gramschmidt(DATA_TYPE* A, DATA_TYPE* R, DATA_TYPE* Q)
{
  int i, j, k;
  DATA_TYPE nrm;
  for (k = 0; k < N; k++) {
    nrm = 0;
    for (i = 0; i < M; i++) {
      nrm += A [i * N + k] * A [i * N + k];
    }

    R [k * N + k] = sqrt(nrm);
    for (i = 0; i < M; i++) {
      Q [i * N + k] = A [i * N + k] / R [k * N + k];
    }

    for (j = k + 1; j < N; j++) {
      R [k * N + j] = 0;
      for (i = 0; i < M; i++) {
        R [k * N + j] += Q [i * N + k] * A [i * N + j];
      }
      for (i = 0; i < M; i++) {
        A [i * N + j] = A [i * N + j] - Q [i * N + k] * R [k * N + j];
      }
    }
  }
}

  __global__ void __generated_kernel_region_0(DATA_TYPE * A,DATA_TYPE  nrm,DATA_TYPE * Q,DATA_TYPE * R);
 
void GPU__gramschmidt(DATA_TYPE* A, DATA_TYPE* R, DATA_TYPE* Q)
{
  int i, j, k;
  DATA_TYPE nrm;

  

	ipmacc_prompt((char*)"IPMACC: memory allocation A\n");
acc_present_or_create((void*)A,(262143+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation R\n");
acc_present_or_create((void*)R,(262143+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation Q\n");
acc_present_or_create((void*)Q,(262143+0)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin A\n");
acc_pcopyin((void*)A,(262143+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin R\n");
acc_pcopyin((void*)R,(262143+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin Q\n");
acc_pcopyin((void*)Q,(262143+0)*sizeof(DATA_TYPE ));


{


  


/* kernel call statement [0, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 0 > gridDim: %d\tblockDim: %d\n",(((abs((int)((N))-(0+0)))/(1)))/256+(((((abs((int)((N))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_0<<<(((abs((int)((N))-(0+0)))/(1)))/256+(((((abs((int)((N))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(DATA_TYPE *)acc_deviceptr((void*)A),
nrm,
(DATA_TYPE *)acc_deviceptr((void*)Q),
(DATA_TYPE *)acc_deviceptr((void*)R));
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
ipmacc_prompt((char*)"IPMACC: memory copyout R\n");
acc_copyout_and_keep((void*)R,(262143+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout Q\n");
acc_copyout_and_keep((void*)Q,(262143+0)*sizeof(DATA_TYPE ));



}

void init_array(DATA_TYPE* A, DATA_TYPE* A2)
{
  int i, j;

  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      A [i * N + j] = ((DATA_TYPE)(i + 1) * (j + 1)) / (M + 1);
      A2 [i * N + j] = A [i * N + j];
    }
  }
}

void compareResults(DATA_TYPE* A, DATA_TYPE* A_outputFromGpu)
{
  int i, j, fail;
  fail = 0;

  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      if (percentDiff(A [i * N + j], A_outputFromGpu [i * N + j]) > PERCENT_DIFF_ERROR_THRESHOLD) {
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
  DATA_TYPE* A_outputFromGpu;
  DATA_TYPE* R;
  DATA_TYPE* Q;

  A = (DATA_TYPE*)malloc(M * N * sizeof(DATA_TYPE));
  A_outputFromGpu = (DATA_TYPE*)malloc(M * N * sizeof(DATA_TYPE));
  R = (DATA_TYPE*)malloc(M * N * sizeof(DATA_TYPE));
  Q = (DATA_TYPE*)malloc(M * N * sizeof(DATA_TYPE));

  fprintf(stdout, "<< Gram-Schmidt decomposition >>\n");

  init_array(A, A_outputFromGpu);

  t_start = rtclock();
  GPU__gramschmidt(A_outputFromGpu, R, Q);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  CPU__gramschmidt(A, R, Q);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(A, A_outputFromGpu);

  free(A);
  free(A_outputFromGpu);
  free(R);
  free(Q);

  return 0;
}



 __global__ void __generated_kernel_region_0(DATA_TYPE * A,DATA_TYPE  nrm,DATA_TYPE * Q,DATA_TYPE * R){
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
if( k < N)
{
    
    nrm = 0;
for(i = 0; i < M; i++)
{
      nrm += A [i * N + k] * A [i * N + k];
    }
R [k * N + k] = sqrt(nrm);
for(i = 0; i < M; i++)
{
      Q [i * N + k] = A [i * N + k] / R [k * N + k];
    }

for(j = k + 1; j < N; j++)
{
      R [k * N + j] = 0;
for(i = 0; i < M; i++)
{
        R [k * N + j] += Q [i * N + k] * A [i * N + j];
      }

for(i = 0; i < M; i++)
{
        A [i * N + j] = A [i * N + j] - Q [i * N + k] * R [k * N + j];
      }
}
}

}
}
}
//append writeback of scalar variables
}

