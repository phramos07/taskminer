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


# define NI 512
# define NJ 512
# define NK 512
# define NL 512
# define NM 512

# define GPU_DEVICE 1


typedef float DATA_TYPE;

void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D)
{
  int i, j;

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NK; j++) {
      A [i * NK + j] = ((DATA_TYPE)i * j) / NI;
    }
  }

  for (i = 0; i < NK; i++) {
    for (j = 0; j < NJ; j++) {
      B [i * NJ + j] = ((DATA_TYPE)i * (j + 1)) / NJ;
    }
  }

  for (i = 0; i < NJ; i++) {
    for (j = 0; j < NM; j++) {
      C [i * NM + j] = ((DATA_TYPE)i * (j + 3)) / NL;
    }
  }

  for (i = 0; i < NM; i++) {
    for (j = 0; j < NL; j++) {
      D [i * NL + j] = ((DATA_TYPE)i * (j + 2)) / NK;
    }
  }
}

void compareResults(DATA_TYPE *G, DATA_TYPE *G_outputFromGpu)
{
  int i, j, fail;
  fail = 0;

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NL; j++) {
      if (percentDiff(G [i * NL + j], G_outputFromGpu [i * NL + j]) > PERCENT_DIFF_ERROR_THRESHOLD) {
        fail++;
      }
    }
  }

  
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void CPU__mm3(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D, DATA_TYPE *E, DATA_TYPE *F, DATA_TYPE *G)
{
  int i, j, k;

  
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      E [i * NJ + j] = 0;
      for (k = 0; k < NK; ++k) {
        E [i * NJ + j] += A [i * NK + k] * B [k * NJ + j];
      }
    }
  }

  
  for (i = 0; i < NJ; i++) {
    for (j = 0; j < NL; j++) {
      F [i * NL + j] = 0;
      for (k = 0; k < NM; ++k) {
        F [i * NL + j] += C [i * NM + k] * D [k * NL + j];
      }
    }
  }

  
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NL; j++) {
      G [i * NL + j] = 0;
      for (k = 0; k < NJ; ++k) {
        G [i * NL + j] += E [i * NJ + k] * F [k * NL + j];
      }
    }
  }
}

  __global__ void __generated_kernel_region_0(DATA_TYPE * A,DATA_TYPE * E,DATA_TYPE * B);
 
  __global__ void __generated_kernel_region_1(DATA_TYPE * F,DATA_TYPE * C,DATA_TYPE * D);
 
  __global__ void __generated_kernel_region_2(DATA_TYPE * G,DATA_TYPE * F,DATA_TYPE * E);
 
void GPU__mm3(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D, DATA_TYPE *E, DATA_TYPE *F, DATA_TYPE *G)
{
  int i, j, k;

  
  

	ipmacc_prompt((char*)"IPMACC: memory allocation A\n");
acc_present_or_create((void*)A,(262143+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation B\n");
acc_present_or_create((void*)B,(262143+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation E\n");
acc_present_or_create((void*)E,(262143+0)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin A\n");
acc_pcopyin((void*)A,(262143+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin B\n");
acc_pcopyin((void*)B,(262143+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin E\n");
acc_pcopyin((void*)E,(262143+0)*sizeof(DATA_TYPE ));


{


  


/* kernel call statement [0, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 0 > gridDim: %d\tblockDim: %d\n",(((abs((int)((NI))-(0+0)))/(1)))/256+(((((abs((int)((NI))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_0<<<(((abs((int)((NI))-(0+0)))/(1)))/256+(((((abs((int)((NI))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(DATA_TYPE *)acc_deviceptr((void*)A),
(DATA_TYPE *)acc_deviceptr((void*)E),
(DATA_TYPE *)acc_deviceptr((void*)B));
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
ipmacc_prompt((char*)"IPMACC: memory copyout E\n");
acc_copyout_and_keep((void*)E,(262143+0)*sizeof(DATA_TYPE ));




  
  

	ipmacc_prompt((char*)"IPMACC: memory allocation C\n");
acc_present_or_create((void*)C,(262143+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation D\n");
acc_present_or_create((void*)D,(262143+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation F\n");
acc_present_or_create((void*)F,(262143+0)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin C\n");
acc_pcopyin((void*)C,(262143+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin D\n");
acc_pcopyin((void*)D,(262143+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin F\n");
acc_pcopyin((void*)F,(262143+0)*sizeof(DATA_TYPE ));


{


  


/* kernel call statement [1, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 1 > gridDim: %d\tblockDim: %d\n",(((abs((int)((NJ))-(0+0)))/(1)))/256+(((((abs((int)((NJ))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_1<<<(((abs((int)((NJ))-(0+0)))/(1)))/256+(((((abs((int)((NJ))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(DATA_TYPE *)acc_deviceptr((void*)F),
(DATA_TYPE *)acc_deviceptr((void*)C),
(DATA_TYPE *)acc_deviceptr((void*)D));
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
acc_copyout_and_keep((void*)C,(262143+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout D\n");
acc_copyout_and_keep((void*)D,(262143+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout F\n");
acc_copyout_and_keep((void*)F,(262143+0)*sizeof(DATA_TYPE ));




  
  

	ipmacc_prompt((char*)"IPMACC: memory allocation E\n");
acc_present_or_create((void*)E,(262143+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation F\n");
acc_present_or_create((void*)F,(262143+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation G\n");
acc_present_or_create((void*)G,(262143+0)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin E\n");
acc_pcopyin((void*)E,(262143+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin F\n");
acc_pcopyin((void*)F,(262143+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin G\n");
acc_pcopyin((void*)G,(262143+0)*sizeof(DATA_TYPE ));


{


  


/* kernel call statement [2, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 2 > gridDim: %d\tblockDim: %d\n",(((abs((int)((NI))-(0+0)))/(1)))/256+(((((abs((int)((NI))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_2<<<(((abs((int)((NI))-(0+0)))/(1)))/256+(((((abs((int)((NI))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(DATA_TYPE *)acc_deviceptr((void*)G),
(DATA_TYPE *)acc_deviceptr((void*)F),
(DATA_TYPE *)acc_deviceptr((void*)E));
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
	ipmacc_prompt((char*)"IPMACC: memory copyout E\n");
acc_copyout_and_keep((void*)E,(262143+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout F\n");
acc_copyout_and_keep((void*)F,(262143+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout G\n");
acc_copyout_and_keep((void*)G,(262143+0)*sizeof(DATA_TYPE ));



}

int main(int argc, char** argv)
{
  double t_start, t_end;

  DATA_TYPE* A;
  DATA_TYPE* B;
  DATA_TYPE* C;
  DATA_TYPE* D;
  DATA_TYPE* E;
  DATA_TYPE* F;
  DATA_TYPE* G;
  DATA_TYPE* G_outputFromGpu;

  A = (DATA_TYPE*)malloc(NI * NK * sizeof(DATA_TYPE));
  B = (DATA_TYPE*)malloc(NK * NJ * sizeof(DATA_TYPE));
  C = (DATA_TYPE*)malloc(NJ * NM * sizeof(DATA_TYPE));
  D = (DATA_TYPE*)malloc(NM * NL * sizeof(DATA_TYPE));
  E = (DATA_TYPE*)malloc(NI * NJ * sizeof(DATA_TYPE));
  F = (DATA_TYPE*)malloc(NJ * NL * sizeof(DATA_TYPE));
  G = (DATA_TYPE*)malloc(NI * NL * sizeof(DATA_TYPE));
  G_outputFromGpu = (DATA_TYPE*)malloc(NI * NL * sizeof(DATA_TYPE));

  fprintf(stdout, "<< Linear Algebra: 3 Matrix Multiplications (E=A.B; F=C.D; G=E.F) >>\n");

  init_array(A, B, C, D);

  t_start = rtclock();
  GPU__mm3(A, B, C, D, E, F, G_outputFromGpu);
  t_end = rtclock();

  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  CPU__mm3(A, B, C, D, E, F, G);
  t_end = rtclock();

  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(G, G_outputFromGpu);

  free(A);
  free(B);
  free(C);
  free(D);
  free(E);
  free(F);
  free(G);
  free(G_outputFromGpu);

  return 0;
}



 __global__ void __generated_kernel_region_0(DATA_TYPE * A,DATA_TYPE * E,DATA_TYPE * B){
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
      E [i * NJ + j] = 0;
for(k = 0; k < NK; ++k)
{
        E [i * NJ + j] += A [i * NK + k] * B [k * NJ + j];
      }
}
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_1(DATA_TYPE * F,DATA_TYPE * C,DATA_TYPE * D){
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
if( i < NJ)
{
for(j = 0; j < NL; j++)
{
      F [i * NL + j] = 0;
for(k = 0; k < NM; ++k)
{
        F [i * NL + j] += C [i * NM + k] * D [k * NL + j];
      }
}
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_2(DATA_TYPE * G,DATA_TYPE * F,DATA_TYPE * E){
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
for(j = 0; j < NL; j++)
{
      G [i * NL + j] = 0;
for(k = 0; k < NJ; ++k)
{
        G [i * NL + j] += E [i * NJ + k] * F [k * NL + j];
      }
}
}

}
}
}
//append writeback of scalar variables
}

