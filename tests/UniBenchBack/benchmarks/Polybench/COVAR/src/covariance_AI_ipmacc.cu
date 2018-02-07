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


#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

#define GPU_DEVICE 1


#define M 2048
#define N 2048

#define sqrt_of_array_cell(x, j) sqrt(x [j])

#define FLOAT_N 3214212.01
#define EPS 0.005


typedef float DATA_TYPE;

void init_arrays(DATA_TYPE* data)
{
  int i, j;

  for (i = 1; i < (M + 1); i++) {
    for (j = 1; j < (N + 1); j++) {
      data [i * (N + 1) + j] = ((DATA_TYPE)i * j) / M;
    }
  }
}

void compareResults(DATA_TYPE* symmat, DATA_TYPE* symmat_outputFromGpu)
{
  int i, j, fail;
  fail = 0;

  for (i = 1; i < (M + 1); i++) {
    for (j = 1; j < (N + 1); j++) {
      if (percentDiff(symmat [i * (N + 1) + j], symmat_outputFromGpu [i * (N + 1) + j]) > PERCENT_DIFF_ERROR_THRESHOLD) {
        fail++;
      }
    }
  }
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void CPU__covariance(DATA_TYPE* data, DATA_TYPE* symmat, DATA_TYPE* mean)
{
  int i, j, j1, j2;

  
  for (j = 1; j < (M + 1); j++) {
    mean [j] = 0.0;
    for (i = 1; i < (N + 1); i++) {
      mean [j] += data [i * (M + 1) + j];
    }
    mean [j] /= FLOAT_N;
  }

  
  for (i = 1; i < (N + 1); i++) {
    for (j = 1; j < (M + 1); j++) {
      data [i * (M + 1) + j] -= mean [j];
    }
  }

  
  for (j1 = 1; j1 < (M + 1); j1++) {
    for (j2 = j1; j2 < (M + 1); j2++) {
      symmat [j1 * (M + 1) + j2] = 0.0;
      for (i = 1; i < N + 1; i++) {
        symmat [j1 * (M + 1) + j2] += data [i * (M + 1) + j1] * data [i * (M + 1) + j2];
      }
      symmat [j2 * (M + 1) + j1] = symmat [j1 * (M + 1) + j2];
    }
  }
}

  __global__ void __generated_kernel_region_0(DATA_TYPE * data,DATA_TYPE * mean);
 
  __global__ void __generated_kernel_region_1(DATA_TYPE * data,DATA_TYPE * mean);
 
  __global__ void __generated_kernel_region_2(DATA_TYPE * symmat,DATA_TYPE * data);
 
void GPU__covariance(DATA_TYPE* data, DATA_TYPE* symmat, DATA_TYPE* mean)
{
  int i, j, j1, j2;

  

  

	ipmacc_prompt((char*)"IPMACC: memory allocation data\n");
acc_present_or_create((void*)data,(4196350+2050)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation mean\n");
acc_present_or_create((void*)mean,(2047+1)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin data\n");
acc_pcopyin((void*)data,(4196350+2050)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin mean\n");
acc_pcopyin((void*)mean,(2047+1)*sizeof(DATA_TYPE ));


{


  


/* kernel call statement [0, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 0 > gridDim: %d\tblockDim: %d\n",(((abs((int)(((M+1)))-(1+0)))/(1)))/256+(((((abs((int)(((M+1)))-(1+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_0<<<(((abs((int)(((M+1)))-(1+0)))/(1)))/256+(((((abs((int)(((M+1)))-(1+0)))/(1)))%(256))==0?0:1),256>>>(
(DATA_TYPE *)acc_deviceptr((void*)data),
(DATA_TYPE *)acc_deviceptr((void*)mean));
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
	ipmacc_prompt((char*)"IPMACC: memory copyout data\n");
acc_copyout_and_keep((void*)data,(4196350+2050)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout mean\n");
acc_copyout_and_keep((void*)mean,(2047+1)*sizeof(DATA_TYPE ));




  
  

	ipmacc_prompt((char*)"IPMACC: memory allocation data\n");
acc_present_or_create((void*)data,(4196350+2050)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation mean\n");
acc_present_or_create((void*)mean,(2047+1)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin data\n");
acc_pcopyin((void*)data,(4196350+2050)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin mean\n");
acc_pcopyin((void*)mean,(2047+1)*sizeof(DATA_TYPE ));


{


  


/* kernel call statement [1, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 1 > gridDim: %d\tblockDim: %d\n",(((abs((int)(((N+1)))-(1+0)))/(1)))/256+(((((abs((int)(((N+1)))-(1+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_1<<<(((abs((int)(((N+1)))-(1+0)))/(1)))/256+(((((abs((int)(((N+1)))-(1+0)))/(1)))%(256))==0?0:1),256>>>(
(DATA_TYPE *)acc_deviceptr((void*)data),
(DATA_TYPE *)acc_deviceptr((void*)mean));
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
	ipmacc_prompt((char*)"IPMACC: memory copyout data\n");
acc_copyout_and_keep((void*)data,(4196350+2050)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout mean\n");
acc_copyout_and_keep((void*)mean,(2047+1)*sizeof(DATA_TYPE ));




  
  

	ipmacc_prompt((char*)"IPMACC: memory allocation data\n");
acc_present_or_create((void*)data,(4196350+2050)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation symmat\n");
acc_present_or_create((void*)symmat,(4196350+2050)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin data\n");
acc_pcopyin((void*)data,(4196350+2050)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin symmat\n");
acc_pcopyin((void*)symmat,(4196350+2050)*sizeof(DATA_TYPE ));


{


  


/* kernel call statement [2, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 2 > gridDim: %d\tblockDim: %d\n",(((abs((int)(((M+1)))-(1+0)))/(1)))/256+(((((abs((int)(((M+1)))-(1+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_2<<<(((abs((int)(((M+1)))-(1+0)))/(1)))/256+(((((abs((int)(((M+1)))-(1+0)))/(1)))%(256))==0?0:1),256>>>(
(DATA_TYPE *)acc_deviceptr((void*)symmat),
(DATA_TYPE *)acc_deviceptr((void*)data));
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
	ipmacc_prompt((char*)"IPMACC: memory copyout data\n");
acc_copyout_and_keep((void*)data,(4196350+2050)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout symmat\n");
acc_copyout_and_keep((void*)symmat,(4196350+2050)*sizeof(DATA_TYPE ));



}

int main()
{
  double t_start, t_end;

  DATA_TYPE* data;
  DATA_TYPE* symmat;
  DATA_TYPE* mean;
  DATA_TYPE* symmat_outputFromGpu;

  data = (DATA_TYPE*)malloc((M + 1) * (N + 1) * sizeof(DATA_TYPE));
  symmat = (DATA_TYPE*)malloc((M + 1) * (M + 1) * sizeof(DATA_TYPE));
  mean = (DATA_TYPE*)malloc((M + 1) * sizeof(DATA_TYPE));
  symmat_outputFromGpu = (DATA_TYPE*)malloc((M + 1) * (M + 1) * sizeof(DATA_TYPE));

  fprintf(stdout, "<< Covariance Computation >>\n");

  init_arrays(data);

  t_start = rtclock();
  GPU__covariance(data, symmat_outputFromGpu, mean);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  init_arrays(data);

  t_start = rtclock();
  CPU__covariance(data, symmat, mean);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(symmat, symmat_outputFromGpu);

  free(data);
  free(symmat);
  free(mean);
  free(symmat_outputFromGpu);

  return 0;
}



 __global__ void __generated_kernel_region_0(DATA_TYPE * data,DATA_TYPE * mean){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  j;
{
{
{
 j=1+(__kernel_getuid_x);
if( j < (M + 1))
{
    mean [j] = 0.0;
for(i = 1; i < (N + 1); i++)
{
      mean [j] += data [i * (M + 1) + j];
    }
mean [j] /= FLOAT_N;
  }

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_1(DATA_TYPE * data,DATA_TYPE * mean){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  j;
{
{
{
 i=1+(__kernel_getuid_x);
if( i < (N + 1))
{
for(j = 1; j < (M + 1); j++)
{
      data [i * (M + 1) + j] -= mean [j];
    }
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_2(DATA_TYPE * symmat,DATA_TYPE * data){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  j1;
int  j2;
{
{
{
 j1=1+(__kernel_getuid_x);
if( j1 < (M + 1))
{
for(j2 = j1; j2 < (M + 1); j2++)
{
      symmat [j1 * (M + 1) + j2] = 0.0;
for(i = 1; i < N + 1; i++)
{
        symmat [j1 * (M + 1) + j2] += data [i * (M + 1) + j1] * data [i * (M + 1) + j2];
      }
symmat [j2 * (M + 1) + j1] = symmat [j1 * (M + 1) + j2];
    }
}

}
}
}
//append writeback of scalar variables
}

