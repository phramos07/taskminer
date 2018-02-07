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


#define PERCENT_DIFF_ERROR_THRESHOLD 10.05

#define GPU_DEVICE 1


#define tmax 500
#define NX 2048
#define NY 2048


typedef float DATA_TYPE;

void init_arrays(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
  int i, j;

  for (i = 0; i < tmax; i++) {
    _fict_ [i] = (DATA_TYPE)i;
  }

  for (i = 0; i < NX; i++) {
    for (j = 0; j < NY; j++) {
      ex [i * NY + j] = ((DATA_TYPE)i * (j + 1) + 1) / NX;
      ey [i * NY + j] = ((DATA_TYPE)(i - 1) * (j + 2) + 2) / NX;
      hz [i * NY + j] = ((DATA_TYPE)(i - 9) * (j + 4) + 3) / NX;
    }
  }
}

void init_array_hz(DATA_TYPE* hz)
{
  int i, j;

  for (i = 0; i < NX; i++) {
    for (j = 0; j < NY; j++) {
      hz [i * NY + j] = ((DATA_TYPE)(i - 9) * (j + 4) + 3) / NX;
    }
  }
}

void compareResults(DATA_TYPE* hz1, DATA_TYPE* hz2)
{
  int i, j, fail;
  fail = 0;

  for (i = 0; i < NX; i++) {
    for (j = 0; j < NY; j++) {
      if (percentDiff(hz1 [i * NY + j], hz2 [i * NY + j]) > PERCENT_DIFF_ERROR_THRESHOLD) {
        fail++;
      }
    }
  }

  
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void CPU__runFdtd(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
  int t, i, j;

  for (t = 0; t < tmax; t++) {
    for (j = 0; j < NY; j++) {
      ey [0 * NY + j] = _fict_ [t];
    }

    for (i = 1; i < NX; i++) {
      for (j = 0; j < NY; j++) {
        ey [i * NY + j] = ey [i * NY + j] - 0.5 * (hz [i * NY + j] - hz [(i - 1) * NY + j]);
      }
    }

    for (i = 0; i < NX; i++) {
      for (j = 1; j < NY; j++) {
        ex [i * (NY + 1) + j] = ex [i * (NY + 1) + j] - 0.5 * (hz [i * NY + j] - hz [i * NY + (j - 1)]);
      }
    }

    for (i = 0; i < NX; i++) {
      for (j = 0; j < NY; j++) {
        hz [i * NY + j] = hz [i * NY + j] - 0.7 * (ex [i * (NY + 1) + (j + 1)] - ex [i * (NY + 1) + j] + ey [(i + 1) * NY + j] - ey [i * NY + j]);
      }
    }
  }
}

  __global__ void __generated_kernel_region_0(DATA_TYPE * hz,DATA_TYPE * _fict_,DATA_TYPE * ey,DATA_TYPE * ex);
 
void GPU__runFdtd(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
  int t, i, j;

    

	ipmacc_prompt((char*)"IPMACC: memory allocation _fict_\n");
acc_present_or_create((void*)_fict_,(499+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation ex\n");
acc_present_or_create((void*)ex,(4196351+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation ey\n");
acc_present_or_create((void*)ey,(4196351+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation hz\n");
acc_present_or_create((void*)hz,(4194303+0)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin _fict_\n");
acc_pcopyin((void*)_fict_,(499+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin ex\n");
acc_pcopyin((void*)ex,(4196351+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin ey\n");
acc_pcopyin((void*)ey,(4196351+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin hz\n");
acc_pcopyin((void*)hz,(4194303+0)*sizeof(DATA_TYPE ));


{


    


/* kernel call statement [0, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 0 > gridDim: %d\tblockDim: %d\n",(((abs((int)((tmax))-(0+0)))/(1)))/256+(((((abs((int)((tmax))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_0<<<(((abs((int)((tmax))-(0+0)))/(1)))/256+(((((abs((int)((tmax))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(DATA_TYPE *)acc_deviceptr((void*)hz),
(DATA_TYPE *)acc_deviceptr((void*)_fict_),
(DATA_TYPE *)acc_deviceptr((void*)ey),
(DATA_TYPE *)acc_deviceptr((void*)ex));
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
	ipmacc_prompt((char*)"IPMACC: memory copyout _fict_\n");
acc_copyout_and_keep((void*)_fict_,(499+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout ex\n");
acc_copyout_and_keep((void*)ex,(4196351+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout ey\n");
acc_copyout_and_keep((void*)ey,(4196351+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout hz\n");
acc_copyout_and_keep((void*)hz,(4194303+0)*sizeof(DATA_TYPE ));



}

int main()
{
  double t_start, t_end;

  DATA_TYPE* _fict_;
  DATA_TYPE* ex;
  DATA_TYPE* ey;
  DATA_TYPE* hz;
  DATA_TYPE* hz_outputFromGpu;

  _fict_ = (DATA_TYPE*)malloc(tmax * sizeof(DATA_TYPE));
  ex = (DATA_TYPE*)malloc(NX * (NY + 1) * sizeof(DATA_TYPE));
  ey = (DATA_TYPE*)malloc((NX + 1) * NY * sizeof(DATA_TYPE));
  hz = (DATA_TYPE*)malloc(NX * NY * sizeof(DATA_TYPE));
  hz_outputFromGpu = (DATA_TYPE*)malloc(NX * NY * sizeof(DATA_TYPE));

  fprintf(stdout, "<< 2-D Finite Different Time Domain Kernel >>\n");

  init_arrays(_fict_, ex, ey, hz);
  init_array_hz(hz_outputFromGpu);

  t_start = rtclock();
  GPU__runFdtd(_fict_, ex, ey, hz_outputFromGpu);
  t_end = rtclock();

  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  CPU__runFdtd(_fict_, ex, ey, hz);
  t_end = rtclock();

  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(hz, hz_outputFromGpu);

  free(_fict_);
  free(ex);
  free(ey);
  free(hz);
  free(hz_outputFromGpu);

  return 0;
}



 __global__ void __generated_kernel_region_0(DATA_TYPE * hz,DATA_TYPE * _fict_,DATA_TYPE * ey,DATA_TYPE * ex){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  j;
int  t;
{
{
{
 t=0+(__kernel_getuid_x);
if( t < tmax)
{
for(j = 0; j < NY; j++)
{
      ey [0 * NY + j] = _fict_ [t];
    }

for(i = 1; i < NX; i++)
{
for(j = 0; j < NY; j++)
{
        ey [i * NY + j] = ey [i * NY + j] - 0.5 * (hz [i * NY + j] - hz [(i - 1) * NY + j]);
      }
}

for(i = 0; i < NX; i++)
{
for(j = 1; j < NY; j++)
{
        ex [i * (NY + 1) + j] = ex [i * (NY + 1) + j] - 0.5 * (hz [i * NY + j] - hz [i * NY + (j - 1)]);
      }
}

for(i = 0; i < NX; i++)
{
for(j = 0; j < NY; j++)
{
        hz [i * NY + j] = hz [i * NY + j] - 0.7 * (ex [i * (NY + 1) + (j + 1)] - ex [i * (NY + 1) + j] + ey [(i + 1) * NY + j] - ey [i * NY + j]);
      }
}
}

}
}
}
//append writeback of scalar variables
}

