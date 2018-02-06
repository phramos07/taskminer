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


#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 1


#define N 4096


typedef float DATA_TYPE;

void init_array(DATA_TYPE* A, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y1, DATA_TYPE* y2, DATA_TYPE* x1_gpu, DATA_TYPE* x2_gpu)
{
  int i, j;

  for (i = 0; i < N; i++) {
    x1 [i] = ((DATA_TYPE)i) / N;
    x2 [i] = ((DATA_TYPE)i + 1) / N;
    x1_gpu [i] = x1 [i];
    x2_gpu [i] = x2 [i];
    y1 [i] = ((DATA_TYPE)i + 3) / N;
    y2 [i] = ((DATA_TYPE)i + 4) / N;
    for (j = 0; j < N; j++) {
      A [i * N + j] = ((DATA_TYPE)i * j) / N;
    }
  }
}

void CPU__runMvt(DATA_TYPE* a, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y1, DATA_TYPE* y2)
{
  int i, j;

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      x1 [i] = x1 [i] + a [i * N + j] * y1 [j];
    }
  }

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      x2 [i] = x2 [i] + a [j * N + i] * y2 [j];
    }
  }
}

  __global__ void __generated_kernel_region_0(DATA_TYPE * a,DATA_TYPE * y1,DATA_TYPE * x1);
 
  __global__ void __generated_kernel_region_1(DATA_TYPE * a,DATA_TYPE * x2,DATA_TYPE * y2);
 
void GPU__runMvt(DATA_TYPE* a, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y1, DATA_TYPE* y2)
{
  int i;

  
  

	ipmacc_prompt((char*)"IPMACC: memory allocation a\n");
acc_present_or_create((void*)a,(16777215+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation x1\n");
acc_present_or_create((void*)x1,(4095+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation y1\n");
acc_present_or_create((void*)y1,(4095+0)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin a\n");
acc_pcopyin((void*)a,(16777215+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin x1\n");
acc_pcopyin((void*)x1,(4095+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin y1\n");
acc_pcopyin((void*)y1,(4095+0)*sizeof(DATA_TYPE ));


{


  


/* kernel call statement [0, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 0 > gridDim: %d\tblockDim: %d\n",(((abs((int)((N))-(0+0)))/(1)))/256+(((((abs((int)((N))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_0<<<(((abs((int)((N))-(0+0)))/(1)))/256+(((((abs((int)((N))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(DATA_TYPE *)acc_deviceptr((void*)a),
(DATA_TYPE *)acc_deviceptr((void*)y1),
(DATA_TYPE *)acc_deviceptr((void*)x1));
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
acc_copyout_and_keep((void*)a,(16777215+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout x1\n");
acc_copyout_and_keep((void*)x1,(4095+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout y1\n");
acc_copyout_and_keep((void*)y1,(4095+0)*sizeof(DATA_TYPE ));




  

	ipmacc_prompt((char*)"IPMACC: memory allocation a\n");
acc_present_or_create((void*)a,(16777215+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation x2\n");
acc_present_or_create((void*)x2,(4095+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation y2\n");
acc_present_or_create((void*)y2,(4095+0)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin a\n");
acc_pcopyin((void*)a,(16777215+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin x2\n");
acc_pcopyin((void*)x2,(4095+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin y2\n");
acc_pcopyin((void*)y2,(4095+0)*sizeof(DATA_TYPE ));


{


  


/* kernel call statement [1, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 1 > gridDim: %d\tblockDim: %d\n",(((abs((int)((N))-(0+0)))/(1)))/256+(((((abs((int)((N))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_1<<<(((abs((int)((N))-(0+0)))/(1)))/256+(((((abs((int)((N))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(DATA_TYPE *)acc_deviceptr((void*)a),
(DATA_TYPE *)acc_deviceptr((void*)x2),
(DATA_TYPE *)acc_deviceptr((void*)y2));
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
acc_copyout_and_keep((void*)a,(16777215+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout x2\n");
acc_copyout_and_keep((void*)x2,(4095+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout y2\n");
acc_copyout_and_keep((void*)y2,(4095+0)*sizeof(DATA_TYPE ));



}

void compareResults(DATA_TYPE* x1, DATA_TYPE* x1_outputFromGpu, DATA_TYPE* x2, DATA_TYPE* x2_outputFromGpu)
{
  int i, fail;
  fail = 0;

  for (i = 0; i < N; i++) {
    if (percentDiff(x1 [i], x1_outputFromGpu [i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
      fail++;
    }

    if (percentDiff(x2 [i], x2_outputFromGpu [i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
      fail++;
    }
  }

  
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

int main()
{
  double t_start, t_end;

  DATA_TYPE* a;
  DATA_TYPE* x1;
  DATA_TYPE* x2;
  DATA_TYPE* x1_outputFromGpu;
  DATA_TYPE* x2_outputFromGpu;
  DATA_TYPE* y_1;
  DATA_TYPE* y_2;

  a = (DATA_TYPE*)malloc(N * N * sizeof(DATA_TYPE));
  x1 = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
  x2 = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
  x1_outputFromGpu = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
  x2_outputFromGpu = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
  y_1 = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
  y_2 = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));

  fprintf(stdout, "<< Matrix Vector Product and Transpose >>\n");

  init_array(a, x1, x2, y_1, y_2, x1_outputFromGpu, x2_outputFromGpu);

  t_start = rtclock();
  GPU__runMvt(a, x1_outputFromGpu, x2_outputFromGpu, y_1, y_2);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  
  CPU__runMvt(a, x1, x2, y_1, y_2);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(x1, x1_outputFromGpu, x2, x2_outputFromGpu);

  free(a);
  free(x1);
  free(x2);
  free(x1_outputFromGpu);
  free(x2_outputFromGpu);
  free(y_1);
  free(y_2);

  return 0;
}



 __global__ void __generated_kernel_region_0(DATA_TYPE * a,DATA_TYPE * y1,DATA_TYPE * x1){
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
    int j;
for(j = 0; j < N; j++)
{
      x1 [i] = x1 [i] + a [i * N + j] * y1 [j];
    }
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_1(DATA_TYPE * a,DATA_TYPE * x2,DATA_TYPE * y2){
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
    int j;
for(j = 0; j < N; j++)
{
      x2 [i] = x2 [i] + a [j * N + i] * y2 [j];
    }
}

}
}
}
//append writeback of scalar variables
}

