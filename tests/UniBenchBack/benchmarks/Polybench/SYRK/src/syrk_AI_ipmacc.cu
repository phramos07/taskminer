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


#define ERROR_THRESHOLD 0.05
#define GPU_DEVICE 1


#define N 1024
#define M 1024



#define alpha 12435
#define beta 4546


typedef float DATA_TYPE;

 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_0(DATA_TYPE * A,DATA_TYPE * D,DATA_TYPE * C);
 
void init_arrays(DATA_TYPE* A, DATA_TYPE* C, DATA_TYPE* D)
{
  int i, j;

  

	ipmacc_prompt((char*)"IPMACC: memory allocation D\n");
acc_present_or_create((void*)D,(1048575+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation C\n");
acc_present_or_create((void*)C,(1048575+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation A\n");
acc_present_or_create((void*)A,(1048575+0)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin D\n");
acc_pcopyin((void*)D,(1048575+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin C\n");
acc_pcopyin((void*)C,(1048575+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin A\n");
acc_pcopyin((void*)A,(1048575+0)*sizeof(DATA_TYPE ));


{


  


/* kernel call statement [0, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 0 > gridDim: %d\tblockDim: %d\n",(((abs((int)((N))-(0+0)))/(1)))/256+(((((abs((int)((N))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_0<<<(((abs((int)((N))-(0+0)))/(1)))/256+(((((abs((int)((N))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(DATA_TYPE *)acc_deviceptr((void*)A),
(DATA_TYPE *)acc_deviceptr((void*)D),
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
	ipmacc_prompt((char*)"IPMACC: memory copyout D\n");
acc_copyout_and_keep((void*)D,(1048575+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout C\n");
acc_copyout_and_keep((void*)C,(1048575+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout A\n");
acc_copyout_and_keep((void*)A,(1048575+0)*sizeof(DATA_TYPE ));



}

 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_1(DATA_TYPE * D,DATA_TYPE * C,int  fail);
 
void compareResults(DATA_TYPE* C, DATA_TYPE* D)
{
  int i, j, fail;
  fail = 0;

  
  

	ipmacc_prompt((char*)"IPMACC: memory allocation C\n");
acc_present_or_create((void*)C,(1048575+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation D\n");
acc_present_or_create((void*)D,(1048575+0)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin C\n");
acc_pcopyin((void*)C,(1048575+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin D\n");
acc_pcopyin((void*)D,(1048575+0)*sizeof(DATA_TYPE ));


{


  


/* kernel call statement [1, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 1 > gridDim: %d\tblockDim: %d\n",(((abs((int)((N))-(0+0)))/(1)))/256+(((((abs((int)((N))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_1<<<(((abs((int)((N))-(0+0)))/(1)))/256+(((((abs((int)((N))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(DATA_TYPE *)acc_deviceptr((void*)D),
(DATA_TYPE *)acc_deviceptr((void*)C),
fail);
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




  
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", ERROR_THRESHOLD, fail);
}

 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_2(DATA_TYPE * C);
 
 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_3(DATA_TYPE * A,DATA_TYPE * C);
 
void syrk(DATA_TYPE* A, DATA_TYPE* C)
{
  int i, j, k;

  

	ipmacc_prompt((char*)"IPMACC: memory allocation C\n");
acc_present_or_create((void*)C,(1048575+0)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin C\n");
acc_pcopyin((void*)C,(1048575+0)*sizeof(DATA_TYPE ));


{


  


/* kernel call statement [2, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 2 > gridDim: %d\tblockDim: %d\n",(((abs((int)((N))-(0+0)))/(1)))/256+(((((abs((int)((N))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_2<<<(((abs((int)((N))-(0+0)))/(1)))/256+(((((abs((int)((N))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
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




  

	ipmacc_prompt((char*)"IPMACC: memory allocation C\n");
acc_present_or_create((void*)C,(1048575+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation A\n");
acc_present_or_create((void*)A,(1048575+0)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin C\n");
acc_pcopyin((void*)C,(1048575+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin A\n");
acc_pcopyin((void*)A,(1048575+0)*sizeof(DATA_TYPE ));


{


  


/* kernel call statement [3, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 3 > gridDim: %d\tblockDim: %d\n",(((abs((int)((N))-(0+0)))/(1)))/256+(((((abs((int)((N))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_3<<<(((abs((int)((N))-(0+0)))/(1)))/256+(((((abs((int)((N))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(DATA_TYPE *)acc_deviceptr((void*)A),
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
ipmacc_prompt((char*)"IPMACC: memory copyout A\n");
acc_copyout_and_keep((void*)A,(1048575+0)*sizeof(DATA_TYPE ));



}

 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_4(DATA_TYPE * D);
 
 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_5(DATA_TYPE * A,DATA_TYPE * D);
 
void syrkGPU(DATA_TYPE* A, DATA_TYPE* D)
{
  int i, j;
  double t_start, t_end;

  t_start = rtclock();

  #pragma omp target  device (GPU_DEVICE)
  #pragma omp target map(to: A[:N*M]) map(tofrom: D[:N*M])
  {
    #pragma omp parallel for
    

	ipmacc_prompt((char*)"IPMACC: memory allocation D\n");
acc_present_or_create((void*)D,(1048575+0)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin D\n");
acc_pcopyin((void*)D,(1048575+0)*sizeof(DATA_TYPE ));


{


    


/* kernel call statement [4, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 4 > gridDim: %d\tblockDim: %d\n",(((abs((int)((N))-(0+0)))/(1)))/256+(((((abs((int)((N))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_4<<<(((abs((int)((N))-(0+0)))/(1)))/256+(((((abs((int)((N))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
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
	ipmacc_prompt((char*)"IPMACC: memory copyout D\n");
acc_copyout_and_keep((void*)D,(1048575+0)*sizeof(DATA_TYPE ));




    #pragma omp parallel for collapse(2)
    

	ipmacc_prompt((char*)"IPMACC: memory allocation A\n");
acc_present_or_create((void*)A,(1048575+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation D\n");
acc_present_or_create((void*)D,(1048575+0)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin A\n");
acc_pcopyin((void*)A,(1048575+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin D\n");
acc_pcopyin((void*)D,(1048575+0)*sizeof(DATA_TYPE ));


{


    


/* kernel call statement [5, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 5 > gridDim: %d\tblockDim: %d\n",(((abs((int)((N))-(0+0)))/(1)))/256+(((((abs((int)((N))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_5<<<(((abs((int)((N))-(0+0)))/(1)))/256+(((((abs((int)((N))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(DATA_TYPE *)acc_deviceptr((void*)A),
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
	ipmacc_prompt((char*)"IPMACC: memory copyout A\n");
acc_copyout_and_keep((void*)A,(1048575+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout D\n");
acc_copyout_and_keep((void*)D,(1048575+0)*sizeof(DATA_TYPE ));



  }

  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
}

int main()
{
  double t_start, t_end;

  DATA_TYPE* A;
  DATA_TYPE* C;
  DATA_TYPE* D;

  A = (DATA_TYPE*)malloc(N * M * sizeof(DATA_TYPE));
  C = (DATA_TYPE*)malloc(N * M * sizeof(DATA_TYPE));
  D = (DATA_TYPE*)malloc(N * M * sizeof(DATA_TYPE));

  fprintf(stdout, "<< Symmetric rank-k operations >>\n");

  init_arrays(A, C, D);
  syrkGPU(A, D);

  t_start = rtclock();
  syrk(A, C);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(C, D);

  free(A);
  free(C);
  free(D);
  return 0;
}


__device__ float __accelerator_absVal( float a ) {
  if ( a  < 0 ) {
   return  ( a  * -1) ; 
 } else
  {
   return  a ; 
 } 
}
__device__ float __accelerator_percentDiff( double val1 , double val2 ) {
  if  ( ( __accelerator_absVal( val1 )  < 0.01) && ( __accelerator_absVal( val2 )  < 0.01) ) {
   return  0.0f ; 
 } else
  {
       return  100.0f * ( __accelerator_absVal( __accelerator_absVal( val1  - val2 )  / __accelerator_absVal( val1  + 0.00000001f ) ) ) ; 
 } 
}

 __global__ void __generated_kernel_region_0(DATA_TYPE * A,DATA_TYPE * D,DATA_TYPE * C){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  j;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < N)
{
for(j = 0; j < M; j++)
{
      A [i * M + j] = ((DATA_TYPE)i * j) / N;
    }

for(j = 0; j < M; j++)
{
      C [i * M + j] = ((DATA_TYPE)i * j + 2) / N;
      D [i * M + j] = ((DATA_TYPE)i * j + 2) / N;
    }
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_1(DATA_TYPE * D,DATA_TYPE * C,int  fail){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  j;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < N)
{
for(j = 0; j < M; j++)
{
      if (__accelerator_percentDiff(C [i * M + j], D [i * M + j]) > ERROR_THRESHOLD) {
        fail++;
      }
    }
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_2(DATA_TYPE * C){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  j;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < N)
{
for(j = 0; j < M; j++)
{
      C [i * M + j] *= beta;
    }
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_3(DATA_TYPE * A,DATA_TYPE * C){
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
if( i < N)
{
for(j = 0; j < M; j++)
{
for(k = 0; k < M; k++)
{
        C [i * N + j] += alpha * A [i * M + k] * A [j * M + k];
      }
}
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_4(DATA_TYPE * D){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  j;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < N)
{
for(j = 0; j < M; j++)
{
        D [i * M + j] *= beta;
      }
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_5(DATA_TYPE * A,DATA_TYPE * D){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  j;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < N)
{
for(j = 0; j < M; j++)
{
        int k;
for(k = 0; k < M; k++)
{
          D [i * M + j] += alpha * A [i * M + k] * A [j * M + k];
        }
}
}

}
}
}
//append writeback of scalar variables
}

