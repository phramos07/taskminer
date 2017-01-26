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

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

  __global__ void __generated_kernel_region_0(int  k,int  n,int * vt,int * v);
 
int main(int argc, char *argv[])
{
  int n = atoi(argv [1]);
  int *v = (int*)malloc(sizeof(int) * n);
  int *vt = (int*)malloc(sizeof(int) * n);
  int i, k = 0;
  long long int AI1 [16];
  AI1 [0] = n + -1;
  AI1 [1] = 4 * AI1 [0];
  AI1 [2] = AI1 [1] / 4;
  AI1 [3] = (AI1 [2] > 0);
  AI1 [4] = (AI1 [3] ? AI1 [2] : 0);
  AI1 [5] = -4 + AI1 [1];
  AI1 [6] = AI1 [0] * AI1 [0];
  AI1 [7] = 4 * AI1 [6];
  AI1 [8] = AI1 [7] * 1;
  AI1 [9] = AI1 [5] > AI1 [8];
  AI1 [10] = (AI1 [9] ? AI1 [5] : AI1 [8]);
  AI1 [11] = AI1 [5] > AI1 [10];
  AI1 [12] = (AI1 [11] ? AI1 [5] : AI1 [10]);
  AI1 [13] = AI1 [12] / 4;
  AI1 [14] = (AI1 [13] > 0);
  AI1 [15] = (AI1 [14] ? AI1 [13] : 0);
  

	ipmacc_prompt((char*)"IPMACC: memory allocation vt\n");
acc_present_or_create((void*)vt,(AI1[4]+0)*sizeof(int ));
ipmacc_prompt((char*)"IPMACC: memory allocation v\n");
acc_present_or_create((void*)v,(AI1[15]+0)*sizeof(int ));
	ipmacc_prompt((char*)"IPMACC: memory copyin vt\n");
acc_pcopyin((void*)vt,(AI1[4]+0)*sizeof(int ));
ipmacc_prompt((char*)"IPMACC: memory copyin v\n");
acc_pcopyin((void*)v,(AI1[15]+0)*sizeof(int ));


{


  


/* kernel call statement [0, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 0 > gridDim: %d\tblockDim: %d\n",(((abs((int)((n))-(0+0)))/(1)))/256+(((((abs((int)((n))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_0<<<(((abs((int)((n))-(0+0)))/(1)))/256+(((((abs((int)((n))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
k,
n,
(int *)acc_deviceptr((void*)vt),
(int *)acc_deviceptr((void*)v));
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
	ipmacc_prompt((char*)"IPMACC: memory copyout vt\n");
acc_copyout_and_keep((void*)vt,(AI1[4]+0)*sizeof(int ));
ipmacc_prompt((char*)"IPMACC: memory copyout v\n");
acc_copyout_and_keep((void*)v,(AI1[15]+0)*sizeof(int ));



  return 0;
}



 __global__ void __generated_kernel_region_0(int  k,int  n,int * vt,int * v){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < n)
{
    if (k == 100) {
      v [k] = vt [i];
    } else{
      v [(i - 1)] += n;
    }
    k += i;
  }

}
}
}
//append writeback of scalar variables
}

