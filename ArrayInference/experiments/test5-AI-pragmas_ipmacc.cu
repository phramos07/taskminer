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

  __global__ void __generated_kernel_region_0(int ** a);
 
int main()
{
  int a [100] [100];
  int i, j;
  acc_create((void*)a, 4040);
  acc_copyin((void*)a, 4040);
  

ipmacc_prompt((char*)"IPMACC: memory allocation a\n");
ipmacc_prompt((char*)"IPMACC: memory copyin a\n");
acc_pcopyin((void*)a,100*100*sizeof(int ));

/* kernel call statement [-1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 0 > gridDim: %d\tblockDim: %d\n",(((abs((int)((10))-(0+0)))/(1)))/256+(((((abs((int)((10))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_0<<<(((abs((int)((10))-(0+0)))/(1)))/256+(((((abs((int)((10))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(int **)acc_deviceptr((void*)a));
}
/* kernel call statement*/
ipmacc_prompt((char*)"IPMACC: memory copyout a\n");
acc_copyout_and_keep((void*)a,100*100*sizeof(int ));
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Synchronizing the region with host\n");
{
cudaError err=cudaDeviceSynchronize();
if(err!=cudaSuccess){
printf("Kernel Launch Error! error code (%d)\n",err);
assert(0&&"Launch Failure!\n");}
}



  acc_copyout_and_keep((void*)a, 4040);
  return 0;
}



 __global__ void __generated_kernel_region_0(int ** a){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  j;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < 10)
{
for(j = 0; j < 10; j++)
{
      a [i] [j] = i * 10 + j;
    }
}

}
}
}
//append writeback of scalar variables
}

