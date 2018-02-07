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
#include <sys/time.h>


#include "../../common/polybenchUtilFuncts.h"


#define PERCENT_DIFF_ERROR_THRESHOLD 0.7


#define NX 8192
#define NY 8192

#define GPU_DEVICE 1

#ifndef M_PI
#define M_PI 3.14159
#endif


typedef float DATA_TYPE;

void init_array(DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *r)
{
  int i, j;

  for (i = 0; i < NX; i++) {
    r [i] = i * M_PI;
    for (j = 0; j < NY; j++) {
      A [i * NY + j] = ((DATA_TYPE)i * j) / NX;
    }
  }

  for (i = 0; i < NY; i++) {
    p [i] = i * M_PI;
  }
}

void compareResults(DATA_TYPE* s, DATA_TYPE* s_outputFromGpu, DATA_TYPE* q, DATA_TYPE* q_outputFromGpu)
{
  int i, fail;
  fail = 0;

  
  for (i = 0; i < NX; i++) {
    if (percentDiff(q [i], q_outputFromGpu [i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
      fail++;
    }
  }

  for (i = 0; i < NY; i++) {
    if (percentDiff(s [i], s_outputFromGpu [i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
      fail++;
    }
  }

  
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void CPU__bicg(DATA_TYPE* A, DATA_TYPE* r, DATA_TYPE* s, DATA_TYPE* p, DATA_TYPE* q)
{
  int i, j;

  for (i = 0; i < NY; i++) {
    s [i] = 0.0;
  }

  for (i = 0; i < NX; i++) {
    q [i] = 0.0;
    for (j = 0; j < NY; j++) {
      
      q [i] = q [i] + A [i * NY + j] * p [j];
    }
  }

  for (j = 0; j < NX; j++) {
    for (i = 0; i < NY; i++) {
      s [j] = s [j] + r [i] * A [i * NY + j];
    }
  }
}

  __global__ void __generated_kernel_region_0(DATA_TYPE * s);
 
  __global__ void __generated_kernel_region_1(DATA_TYPE * A,DATA_TYPE * s,DATA_TYPE * r);
 
  __global__ void __generated_kernel_region_2(DATA_TYPE * A,DATA_TYPE * q,DATA_TYPE * p);
 
void GPU__bicg(DATA_TYPE* A, DATA_TYPE* r, DATA_TYPE* s, DATA_TYPE* p, DATA_TYPE* q)
{
  int i, j;

  

	ipmacc_prompt((char*)"IPMACC: memory allocation s\n");
acc_present_or_create((void*)s,(8191+0)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin s\n");
acc_pcopyin((void*)s,(8191+0)*sizeof(DATA_TYPE ));


{


  


/* kernel call statement [0, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 0 > gridDim: %d\tblockDim: %d\n",(((abs((int)((NY))-(0+0)))/(1)))/256+(((((abs((int)((NY))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_0<<<(((abs((int)((NY))-(0+0)))/(1)))/256+(((((abs((int)((NY))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(DATA_TYPE *)acc_deviceptr((void*)s));
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
	ipmacc_prompt((char*)"IPMACC: memory copyout s\n");
acc_copyout_and_keep((void*)s,(8191+0)*sizeof(DATA_TYPE ));




    

	ipmacc_prompt((char*)"IPMACC: memory allocation A\n");
acc_present_or_create((void*)A,(67108863+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation r\n");
acc_present_or_create((void*)r,(8191+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation s\n");
acc_present_or_create((void*)s,(8191+0)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin A\n");
acc_pcopyin((void*)A,(67108863+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin r\n");
acc_pcopyin((void*)r,(8191+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin s\n");
acc_pcopyin((void*)s,(8191+0)*sizeof(DATA_TYPE ));


{


    


/* kernel call statement [1, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 1 > gridDim: %d\tblockDim: %d\n",(((abs((int)((NY))-(0+0)))/(1)))/256+(((((abs((int)((NY))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_1<<<(((abs((int)((NY))-(0+0)))/(1)))/256+(((((abs((int)((NY))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(DATA_TYPE *)acc_deviceptr((void*)A),
(DATA_TYPE *)acc_deviceptr((void*)s),
(DATA_TYPE *)acc_deviceptr((void*)r));
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
ipmacc_prompt((char*)"IPMACC: memory copyout r\n");
acc_copyout_and_keep((void*)r,(8191+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout s\n");
acc_copyout_and_keep((void*)s,(8191+0)*sizeof(DATA_TYPE ));




     

	ipmacc_prompt((char*)"IPMACC: memory allocation A\n");
acc_present_or_create((void*)A,(67108863+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation p\n");
acc_present_or_create((void*)p,(8191+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation q\n");
acc_present_or_create((void*)q,(8191+0)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin A\n");
acc_pcopyin((void*)A,(67108863+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin p\n");
acc_pcopyin((void*)p,(8191+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin q\n");
acc_pcopyin((void*)q,(8191+0)*sizeof(DATA_TYPE ));


{


     


/* kernel call statement [2, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 2 > gridDim: %d\tblockDim: %d\n",(((abs((int)((NX))-(0+0)))/(1)))/256+(((((abs((int)((NX))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_2<<<(((abs((int)((NX))-(0+0)))/(1)))/256+(((((abs((int)((NX))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(DATA_TYPE *)acc_deviceptr((void*)A),
(DATA_TYPE *)acc_deviceptr((void*)q),
(DATA_TYPE *)acc_deviceptr((void*)p));
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
ipmacc_prompt((char*)"IPMACC: memory copyout p\n");
acc_copyout_and_keep((void*)p,(8191+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout q\n");
acc_copyout_and_keep((void*)q,(8191+0)*sizeof(DATA_TYPE ));



}

int main(int argc, char** argv)
{
  double t_start, t_end;

  DATA_TYPE* A;
  DATA_TYPE* r;
  DATA_TYPE* s;
  DATA_TYPE* p;
  DATA_TYPE* q;
  DATA_TYPE* s_GPU;
  DATA_TYPE* q_GPU;

  A = (DATA_TYPE*)malloc(NX * NY * sizeof(DATA_TYPE));
  r = (DATA_TYPE*)malloc(NX * sizeof(DATA_TYPE));
  s = (DATA_TYPE*)malloc(NY * sizeof(DATA_TYPE));
  p = (DATA_TYPE*)malloc(NY * sizeof(DATA_TYPE));
  q = (DATA_TYPE*)malloc(NX * sizeof(DATA_TYPE));
  s_GPU = (DATA_TYPE*)malloc(NY * sizeof(DATA_TYPE));
  q_GPU = (DATA_TYPE*)malloc(NX * sizeof(DATA_TYPE));

  fprintf(stdout, "<< BiCG Sub Kernel of BiCGStab Linear Solver >>\n");

  init_array(A, p, r);

  t_start = rtclock();
  GPU__bicg(A, r, s_GPU, p, q_GPU);
  t_end = rtclock();

  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  CPU__bicg(A, r, s, p, q);
  t_end = rtclock();

  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(s, s_GPU, q, q_GPU);

  free(A);
  free(r);
  free(s);
  free(p);
  free(q);
  free(s_GPU);
  free(q_GPU);

  return 0;
}



 __global__ void __generated_kernel_region_0(DATA_TYPE * s){
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
    s [i] = 0.0;
  }

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_1(DATA_TYPE * A,DATA_TYPE * s,DATA_TYPE * r){
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
      s [j] = s [j] + r [i] * A [i * NY + j];
    }
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_2(DATA_TYPE * A,DATA_TYPE * q,DATA_TYPE * p){
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
    q [i] = 0.0;
for(j = 0; j < NY; j++)
{
      q [i] = q [i] + A [i * NY + j] * p [j];
    }
}

}
}
}
//append writeback of scalar variables
}

