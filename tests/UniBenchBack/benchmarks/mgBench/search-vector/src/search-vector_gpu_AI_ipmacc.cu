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
#include "../../common/mgbenchUtilFunctions.h"

#define SIZE 1000
#define GPU_DEVICE 0
#define PERCENT_DIFF_ERROR_THRESHOLD 0.01


  __global__ void __generated_kernel_region_0(float * a);
 
void init(float *a)
{
  int i;
  

	ipmacc_prompt((char*)"IPMACC: memory allocation a\n");
acc_present_or_create((void*)a,(999+0)*sizeof(float ));
	ipmacc_prompt((char*)"IPMACC: memory copyin a\n");
acc_pcopyin((void*)a,(999+0)*sizeof(float ));


{


  


/* kernel call statement [0, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 0 > gridDim: %d\tblockDim: %d\n",(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_0<<<(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(float *)acc_deviceptr((void*)a));
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
acc_copyout_and_keep((void*)a,(999+0)*sizeof(float ));



}

int search_GPU(float *a, float c)
{
  int i;
  int find = -1;
  int *find2;

  find2 = &find;

    #pragma omp target device (GPU_DEVICE)
    #pragma omp target map(to: a[:SIZE]) map(from: find2[:1])
  {
        #pragma omp parallel for
    for (i = 0; i < SIZE; ++i) {
      if (a [i] == c) {
        find2 [0] = i;
        i = SIZE;
      }
    }
  }

  return find;
}

int search_CPU(float *a, float c)
{
  int i;
  int find = -1;

  for (i = 0; i < SIZE; ++i) {
    if (a [i] == c) {
      find = i;
      i = SIZE;
    }
  }

  return find;
}

int main(int argc, char *argv[])
{
  double t_start, t_end;
  float *a, c;
  int find_cpu, find_gpu;

  a = (float*)malloc(sizeof(float) * SIZE);
  c = (float)SIZE - 5;

  init(a);

  fprintf(stdout, "<< Search Vector >>\n");

  t_start = rtclock();
  find_gpu = search_GPU(a, c);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  find_cpu = search_CPU(a, c);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  if (find_gpu == find_cpu) {
    printf("Working %d=%d\n", find_gpu, find_cpu);
  } else{
    printf("Error %d != %d\n", find_gpu, find_cpu);
  }

  free(a);

  return 0;
}



 __global__ void __generated_kernel_region_0(float * a){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < SIZE)
{
    a [i] = 2 * i + 7;
  }

}
}
}
//append writeback of scalar variables
}

