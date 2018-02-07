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


#define ERROR_THRESHOLD 1.0

#define GPU_DEVICE 1


# define NI 1024
# define NJ 1024
# define NK 1024
# define NL 1024


typedef float DATA_TYPE;

void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D)
{
  int i, j;

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NK; j++) {
      A [i * NI + j] = ((DATA_TYPE)i * j) / NI;
    }
  }

  for (i = 0; i < NK; i++) {
    for (j = 0; j < NJ; j++) {
      B [i * NK + j] = ((DATA_TYPE)i * (j + 1)) / NJ;
    }
  }

  for (i = 0; i < NL; i++) {
    for (j = 0; j < NJ; j++) {
      C [i * NL + j] = ((DATA_TYPE)i * (j + 3)) / NL;
    }
  }

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NL; j++) {
      D [i * NL + j] = ((DATA_TYPE)i * (j + 2)) / NK;
    }
  }
}

void compareResults(DATA_TYPE *E, DATA_TYPE *E_GPU)
{
  int i, j, fail;
  fail = 0;

  for (i = 0; i < NL; i++) {
    for (j = 0; j < NI; j++) {
      if (percentDiff(E [i * NI + j], E_GPU [i * NI + j]) > ERROR_THRESHOLD) {
        fail++;
      }
    }
  }

  
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", ERROR_THRESHOLD, fail);
}

void CPU__mm2(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E)
{
  int i, j, k;

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      C [i * NJ + j] = 0.0;
      for (k = 0; k < NK; ++k) {
        C [i * NJ + j] += A [i * NK + k] * B [k * NJ + j];
      }
    }
  }

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NL; j++) {
      E [i * NL + j] = 0.0;
      for (k = 0; k < NJ; ++k) {
        E [i * NL + j] += C [i * NJ + k] * D [k * NL + j];
      }
    }
  }
}

  __global__ void __generated_kernel_region_0(DATA_TYPE * A,DATA_TYPE * B,DATA_TYPE * C);
 
  __global__ void __generated_kernel_region_1(DATA_TYPE * D,DATA_TYPE * E,DATA_TYPE * C);
 
void GPU__mm2(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E)
{
  int i, j;

  

	ipmacc_prompt((char*)"IPMACC: memory allocation A\n");
acc_present_or_create((void*)A,(1048575+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation B\n");
acc_present_or_create((void*)B,(1048575+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation C\n");
acc_present_or_create((void*)C,(1048575+0)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin A\n");
acc_pcopyin((void*)A,(1048575+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin B\n");
acc_pcopyin((void*)B,(1048575+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin C\n");
acc_pcopyin((void*)C,(1048575+0)*sizeof(DATA_TYPE ));


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
acc_copyout_and_keep((void*)A,(1048575+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout B\n");
acc_copyout_and_keep((void*)B,(1048575+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout C\n");
acc_copyout_and_keep((void*)C,(1048575+0)*sizeof(DATA_TYPE ));




  

	ipmacc_prompt((char*)"IPMACC: memory allocation C\n");
acc_present_or_create((void*)C,(1048575+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation D\n");
acc_present_or_create((void*)D,(1048575+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation E\n");
acc_present_or_create((void*)E,(1048575+0)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin C\n");
acc_pcopyin((void*)C,(1048575+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin D\n");
acc_pcopyin((void*)D,(1048575+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin E\n");
acc_pcopyin((void*)E,(1048575+0)*sizeof(DATA_TYPE ));


{


  


/* kernel call statement [1, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 1 > gridDim: %d\tblockDim: %d\n",(((abs((int)((NI))-(0+0)))/(1)))/256+(((((abs((int)((NI))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_1<<<(((abs((int)((NI))-(0+0)))/(1)))/256+(((((abs((int)((NI))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(DATA_TYPE *)acc_deviceptr((void*)D),
(DATA_TYPE *)acc_deviceptr((void*)E),
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
acc_copyout_and_keep((void*)C,(1048575+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout D\n");
acc_copyout_and_keep((void*)D,(1048575+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout E\n");
acc_copyout_and_keep((void*)E,(1048575+0)*sizeof(DATA_TYPE ));



}

int main(int argc, char** argv)
{
  double t_start, t_end, t_start_GPU, t_end_GPU;

  DATA_TYPE* C;
  DATA_TYPE* A;
  DATA_TYPE* B;
  DATA_TYPE* D;
  DATA_TYPE* E;
  DATA_TYPE* E_GPU;

  C = (DATA_TYPE*)malloc(NI * NJ * sizeof(DATA_TYPE));
  A = (DATA_TYPE*)malloc(NI * NK * sizeof(DATA_TYPE));
  B = (DATA_TYPE*)malloc(NK * NJ * sizeof(DATA_TYPE));
  D = (DATA_TYPE*)malloc(NJ * NL * sizeof(DATA_TYPE));
  E = (DATA_TYPE*)malloc(NI * NL * sizeof(DATA_TYPE));
  E_GPU = (DATA_TYPE*)malloc(NI * NL * sizeof(DATA_TYPE));

  fprintf(stdout, "<< Linear Algebra: 2 Matrix Multiplications (D=A.B; E=C.D) >>\n");

  init_array(A, B, C, D);

  t_start_GPU = rtclock();
  GPU__mm2(A, B, C, D, E_GPU);
  t_end_GPU = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end_GPU - t_start_GPU);

  t_start = rtclock();
  CPU__mm2(A, B, C, D, E);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(E, E_GPU);

  free(C);
  free(A);
  free(B);
  free(D);
  free(E);
  free(E_GPU);

  return 0;
}



 __global__ void __generated_kernel_region_0(DATA_TYPE * A,DATA_TYPE * B,DATA_TYPE * C){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  j;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < NI)
{
for(j = 0; j < NJ; j++)
{
      C [i * NJ + j] = 0.0;
      int k;
for(k = 0; k < NK; ++k)
{
        C [i * NJ + j] += A [i * NK + k] * B [k * NJ + j];
      }
}
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_1(DATA_TYPE * D,DATA_TYPE * E,DATA_TYPE * C){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  j;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < NI)
{
for(j = 0; j < NL; j++)
{
      E [i * NL + j] = 0.0;
      int k;
for(k = 0; k < NJ; ++k)
{
        E [i * NL + j] += C [i * NJ + k] * D [k * NL + j];
      }
}
}

}
}
}
//append writeback of scalar variables
}

