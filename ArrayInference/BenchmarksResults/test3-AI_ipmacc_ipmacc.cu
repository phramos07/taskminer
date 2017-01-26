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
#include <assert.h>
#include <openacc.h>
#define IPMACC_MAX1(A)   (A)
#define IPMACC_MAX2(A, B) (A > B ? A : B)
#define IPMACC_MAX3(A, B, C) (A > B ? (A > C ? A : (B > C ? B : C)) : (B > C ? C : B))
#ifdef __cplusplus
#endif

#include <cuda.h>

__global__ void __generated_kernel_region_0(int b, int * n);

void func(int a, int b)
{
  int *n;
  n = (int*)malloc(sizeof(int) * (b + 1));


  long long int AI1 [4];
  AI1 [0] = b > 0;
  AI1 [1] = (AI1 [0] ? b : 0);
  AI1 [2] = 4 * AI1 [1];
  AI1 [3] = 4 + AI1 [2];
  acc_create((void*)n, AI1 [3]);
  acc_copyin((void*)n, AI1 [3]);
  {

    {
      if (getenv("IPMACC_VERBOSE")) {
        printf("IPMACC: Launching kernel 0 > gridDim: %d\tblockDim: %d\n", (((abs((int)((b)) - (0 + 0))) / (1))) / 256 + (((((abs((int)((b)) - (0 + 0))) / (1))) % (256)) == 0 ? 0 : 1), 256);
      }
      __generated_kernel_region_0 << < (((abs((int)((b)) - (0 + 0))) / (1))) / 256 + (((((abs((int)((b)) - (0 + 0))) / (1))) % (256)) == 0 ? 0 : 1), 256 >> > (
        b,
        (int*)acc_deviceptr((void*)n));
    }

    if (getenv("IPMACC_VERBOSE")) {
      printf("IPMACC: Synchronizing the region with host\n");
    }
    {
      cudaError err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
        printf("Kernel Launch Error! error code (%d)\n", err);
        assert(0 && "Launch Failure!\n");
      }
    }
  }
  acc_copyout_and_keep((void*)n, AI1 [3]);
}

int main()
{
  func(83, 92);
  return 0;
}

__global__ void __generated_kernel_region_0(int b, int * n)
{
  int __kernel_getuid_x = threadIdx.x + blockIdx.x * blockDim.x;
  int __kernel_getuid_y = threadIdx.y + blockIdx.y * blockDim.y;
  int __kernel_getuid_z = threadIdx.z + blockIdx.z * blockDim.z;
  {
    {
      {
        int i = 0 + (__kernel_getuid_x);
        if (i < b) {
          n [i + 1] = n [i + 1] + 1;
        }
      }
    }
  }

}



