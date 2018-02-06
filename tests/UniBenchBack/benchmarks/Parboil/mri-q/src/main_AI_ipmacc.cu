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
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <malloc.h>

#include "../../common/parboil.h"

#include "file.h"
#include "computeQ.cc"

#include "../../common/polybenchUtilFuncts.h"


#define ERROR_THRESHOLD 0.1
#define GPU_DEVICE 1
double t_start, t_end, t_start_GPU, t_end_GPU;

float *Qr_GPU, *Qi_GPU;   
float *Qr_CPU, *Qi_CPU;   
int N;

typedef float DATA_TYPE;

 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_0(DATA_TYPE * A,DATA_TYPE * A_GPU,int  N,int  fail);
 
 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_1(DATA_TYPE * B,DATA_TYPE * B_GPU,int  N,int  fail);
 
void compareResults(DATA_TYPE *A, DATA_TYPE *A_GPU, DATA_TYPE *B, DATA_TYPE *B_GPU)
{
  int i, fail = 0;

  long long int AI1 [7];
  AI1 [0] = N > 1;
  AI1 [1] = (AI1 [0] ? N : 1);
  AI1 [2] = AI1 [1] + -1;
  AI1 [3] = 4 * AI1 [2];
  AI1 [4] = AI1 [3] / 4;
  AI1 [5] = (AI1 [4] > 0);
  AI1 [6] = (AI1 [5] ? AI1 [4] : 0);
  

	ipmacc_prompt((char*)"IPMACC: memory allocation A\n");
acc_present_or_create((void*)A,(AI1[6]+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation A_GPU\n");
acc_present_or_create((void*)A_GPU,(AI1[6]+0)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin A\n");
acc_pcopyin((void*)A,(AI1[6]+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin A_GPU\n");
acc_pcopyin((void*)A_GPU,(AI1[6]+0)*sizeof(DATA_TYPE ));


{


  


/* kernel call statement [0, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 0 > gridDim: %d\tblockDim: %d\n",(((abs((int)((N))-(0+0)))/(1)))/256+(((((abs((int)((N))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_0<<<(((abs((int)((N))-(0+0)))/(1)))/256+(((((abs((int)((N))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(DATA_TYPE *)acc_deviceptr((void*)A),
(DATA_TYPE *)acc_deviceptr((void*)A_GPU),
N,
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
	ipmacc_prompt((char*)"IPMACC: memory copyout A\n");
acc_copyout_and_keep((void*)A,(AI1[6]+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout A_GPU\n");
acc_copyout_and_keep((void*)A_GPU,(AI1[6]+0)*sizeof(DATA_TYPE ));




  long long int AI2 [7];
  AI2 [0] = N > 1;
  AI2 [1] = (AI2 [0] ? N : 1);
  AI2 [2] = AI2 [1] + -1;
  AI2 [3] = 4 * AI2 [2];
  AI2 [4] = AI2 [3] / 4;
  AI2 [5] = (AI2 [4] > 0);
  AI2 [6] = (AI2 [5] ? AI2 [4] : 0);
  

	ipmacc_prompt((char*)"IPMACC: memory allocation B\n");
acc_present_or_create((void*)B,(AI2[6]+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory allocation B_GPU\n");
acc_present_or_create((void*)B_GPU,(AI2[6]+0)*sizeof(DATA_TYPE ));
	ipmacc_prompt((char*)"IPMACC: memory copyin B\n");
acc_pcopyin((void*)B,(AI2[6]+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyin B_GPU\n");
acc_pcopyin((void*)B_GPU,(AI2[6]+0)*sizeof(DATA_TYPE ));


{


  


/* kernel call statement [1, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 1 > gridDim: %d\tblockDim: %d\n",(((abs((int)((N))-(0+0)))/(1)))/256+(((((abs((int)((N))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_1<<<(((abs((int)((N))-(0+0)))/(1)))/256+(((((abs((int)((N))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(DATA_TYPE *)acc_deviceptr((void*)B),
(DATA_TYPE *)acc_deviceptr((void*)B_GPU),
N,
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
	ipmacc_prompt((char*)"IPMACC: memory copyout B\n");
acc_copyout_and_keep((void*)B,(AI2[6]+0)*sizeof(DATA_TYPE ));
ipmacc_prompt((char*)"IPMACC: memory copyout B_GPU\n");
acc_copyout_and_keep((void*)B_GPU,(AI2[6]+0)*sizeof(DATA_TYPE ));




  
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", ERROR_THRESHOLD, fail);
}

 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_2(float * phiMag,struct kValues * kVals,float * kz,float * kx,float * ky,int  numK);
 
double mriqGPU(int argc, char *argv[])
{
  int numX, numK;   
  int original_numK;    
  float *kx, *ky, *kz;    
  float *x, *y, *z;   
  float *phiR, *phiI;   
  float *phiMag;    
  struct kValues* kVals;

  struct pb_Parameters *params;




  
  params = pb_ReadParameters(&argc, argv);
  if ((params->inpFiles [0] == NULL) || (params->inpFiles [1] != NULL)) {
    fprintf(stderr, "Expecting one input filename\n");
    exit(-1);
  }

  

  inputData(params->inpFiles [0],
            &original_numK, &numX,
            &kx, &ky, &kz,
            &x, &y, &z,
            &phiR, &phiI);

  
  if (argc < 2) {
    numK = original_numK;
  } else{
    int inputK;
    char *end;
    inputK = strtol(argv [1], &end, 10);
    if (end == argv [1]) {
      fprintf(stderr, "Expecting an integer parameter\n");
      exit(-1);
    }

    numK = MIN(inputK, original_numK);
  }






  
  createDataStructsCPU(numK, numX, &phiMag, &Qr_GPU, &Qi_GPU);

  ComputePhiMagCPU(numK, phiR, phiI, phiMag);

  kVals = (struct kValues*)calloc(numK, sizeof(struct kValues));
  int k;
  #pragma omp parallel for
  long long int AI1 [18];
  AI1 [0] = numK + -1;
  AI1 [1] = 4 * AI1 [0];
  AI1 [2] = AI1 [1] / 4;
  AI1 [3] = (AI1 [2] > 0);
  AI1 [4] = (AI1 [3] ? AI1 [2] : 0);
  AI1 [5] = 16 * AI1 [0];
  AI1 [6] = 12 + AI1 [5];
  AI1 [7] = 8 + AI1 [5];
  AI1 [8] = 4 + AI1 [5];
  AI1 [9] = AI1 [8] > AI1 [5];
  AI1 [10] = (AI1 [9] ? AI1 [8] : AI1 [5]);
  AI1 [11] = AI1 [7] > AI1 [10];
  AI1 [12] = (AI1 [11] ? AI1 [7] : AI1 [10]);
  AI1 [13] = AI1 [6] > AI1 [12];
  AI1 [14] = (AI1 [13] ? AI1 [6] : AI1 [12]);
  AI1 [15] = AI1 [14] / 16;
  AI1 [16] = (AI1 [15] > 0);
  AI1 [17] = (AI1 [16] ? AI1 [15] : 0);
  

	ipmacc_prompt((char*)"IPMACC: memory allocation kx\n");
acc_present_or_create((void*)kx,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation ky\n");
acc_present_or_create((void*)ky,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation kz\n");
acc_present_or_create((void*)kz,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation phiMag\n");
acc_present_or_create((void*)phiMag,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation kVals\n");
acc_present_or_create((void*)kVals,(AI1[17]+0)*sizeof(struct kValues ));
	ipmacc_prompt((char*)"IPMACC: memory copyin kx\n");
acc_pcopyin((void*)kx,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin ky\n");
acc_pcopyin((void*)ky,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin kz\n");
acc_pcopyin((void*)kz,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin phiMag\n");
acc_pcopyin((void*)phiMag,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin kVals\n");
acc_pcopyin((void*)kVals,(AI1[17]+0)*sizeof(struct kValues ));


{


  


/* kernel call statement [2, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 2 > gridDim: %d\tblockDim: %d\n",(((abs((int)((numK))-(0+0)))/(1)))/256+(((((abs((int)((numK))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_2<<<(((abs((int)((numK))-(0+0)))/(1)))/256+(((((abs((int)((numK))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(float *)acc_deviceptr((void*)phiMag),
(struct kValues *)acc_deviceptr((void*)kVals),
(float *)acc_deviceptr((void*)kz),
(float *)acc_deviceptr((void*)kx),
(float *)acc_deviceptr((void*)ky),
numK);
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
	ipmacc_prompt((char*)"IPMACC: memory copyout kx\n");
acc_copyout_and_keep((void*)kx,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout ky\n");
acc_copyout_and_keep((void*)ky,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout kz\n");
acc_copyout_and_keep((void*)kz,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout phiMag\n");
acc_copyout_and_keep((void*)phiMag,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout kVals\n");
acc_copyout_and_keep((void*)kVals,(AI1[17]+0)*sizeof(struct kValues ));




  t_start_GPU = rtclock();
  ComputeQGPU(numK, numX, kVals, x, y, z, Qr_GPU, Qi_GPU);
  t_end_GPU = rtclock();



  





  N = numX;

  free(kx);
  free(ky);
  free(kz);
  free(x);
  free(y);
  free(z);
  free(phiR);
  free(phiI);
  free(phiMag);
  free(kVals);

  return t_end_GPU - t_start_GPU;


}

 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_3(float * phiMag,struct kValues * kVals,float * kz,float * kx,float * ky,int  numK);
 
double mriqCPU(int argc, char *argv[])
{
  int numX, numK;   
  int original_numK;    
  float *kx, *ky, *kz;    
  float *x, *y, *z;   
  float *phiR, *phiI;   
  float *phiMag;    
  struct kValues* kVals;

  struct pb_Parameters *params;




  
  params = pb_ReadParameters(&argc, argv);
  if ((params->inpFiles [0] == NULL) || (params->inpFiles [1] != NULL)) {
    fprintf(stderr, "Expecting one input filename\n");
    exit(-1);
  }

  

  inputData(params->inpFiles [0],
            &original_numK, &numX,
            &kx, &ky, &kz,
            &x, &y, &z,
            &phiR, &phiI);

  
  if (argc < 2) {
    numK = original_numK;
  } else{
    int inputK;
    char *end;
    inputK = strtol(argv [1], &end, 10);
    if (end == argv [1]) {
      fprintf(stderr, "Expecting an integer parameter\n");
      exit(-1);
    }

    numK = MIN(inputK, original_numK);
  }






  
  createDataStructsCPU(numK, numX, &phiMag, &Qr_CPU, &Qi_CPU);

  ComputePhiMagCPU(numK, phiR, phiI, phiMag);

  kVals = (struct kValues*)calloc(numK, sizeof(struct kValues));
  int k;
  #pragma omp parallel for
  long long int AI1 [18];
  AI1 [0] = numK + -1;
  AI1 [1] = 4 * AI1 [0];
  AI1 [2] = AI1 [1] / 4;
  AI1 [3] = (AI1 [2] > 0);
  AI1 [4] = (AI1 [3] ? AI1 [2] : 0);
  AI1 [5] = 16 * AI1 [0];
  AI1 [6] = 12 + AI1 [5];
  AI1 [7] = 8 + AI1 [5];
  AI1 [8] = 4 + AI1 [5];
  AI1 [9] = AI1 [8] > AI1 [5];
  AI1 [10] = (AI1 [9] ? AI1 [8] : AI1 [5]);
  AI1 [11] = AI1 [7] > AI1 [10];
  AI1 [12] = (AI1 [11] ? AI1 [7] : AI1 [10]);
  AI1 [13] = AI1 [6] > AI1 [12];
  AI1 [14] = (AI1 [13] ? AI1 [6] : AI1 [12]);
  AI1 [15] = AI1 [14] / 16;
  AI1 [16] = (AI1 [15] > 0);
  AI1 [17] = (AI1 [16] ? AI1 [15] : 0);
  

	ipmacc_prompt((char*)"IPMACC: memory allocation kx\n");
acc_present_or_create((void*)kx,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation ky\n");
acc_present_or_create((void*)ky,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation kz\n");
acc_present_or_create((void*)kz,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation phiMag\n");
acc_present_or_create((void*)phiMag,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation kVals\n");
acc_present_or_create((void*)kVals,(AI1[17]+0)*sizeof(struct kValues ));
	ipmacc_prompt((char*)"IPMACC: memory copyin kx\n");
acc_pcopyin((void*)kx,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin ky\n");
acc_pcopyin((void*)ky,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin kz\n");
acc_pcopyin((void*)kz,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin phiMag\n");
acc_pcopyin((void*)phiMag,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin kVals\n");
acc_pcopyin((void*)kVals,(AI1[17]+0)*sizeof(struct kValues ));


{


  


/* kernel call statement [3, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 3 > gridDim: %d\tblockDim: %d\n",(((abs((int)((numK))-(0+0)))/(1)))/256+(((((abs((int)((numK))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_3<<<(((abs((int)((numK))-(0+0)))/(1)))/256+(((((abs((int)((numK))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(float *)acc_deviceptr((void*)phiMag),
(struct kValues *)acc_deviceptr((void*)kVals),
(float *)acc_deviceptr((void*)kz),
(float *)acc_deviceptr((void*)kx),
(float *)acc_deviceptr((void*)ky),
numK);
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
	ipmacc_prompt((char*)"IPMACC: memory copyout kx\n");
acc_copyout_and_keep((void*)kx,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout ky\n");
acc_copyout_and_keep((void*)ky,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout kz\n");
acc_copyout_and_keep((void*)kz,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout phiMag\n");
acc_copyout_and_keep((void*)phiMag,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout kVals\n");
acc_copyout_and_keep((void*)kVals,(AI1[17]+0)*sizeof(struct kValues ));




  t_start = rtclock();
  ComputeQCPU(numK, numX, kVals, x, y, z, Qr_CPU, Qi_CPU);
  t_end = rtclock();



  





  N = numX;

  free(kx);
  free(ky);
  free(kz);
  free(x);
  free(y);
  free(z);
  free(phiR);
  free(phiI);
  free(phiMag);
  free(kVals);

  return t_end - t_start;


}

int main(int argc, char *argv[])
{
  double t_GPU, t_CPU;

  t_GPU = mriqGPU(argc, argv);
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_GPU);

  t_CPU = mriqCPU(argc, argv);
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_CPU);

  compareResults(Qr_CPU, Qr_GPU, Qi_CPU, Qi_GPU);

  free(Qr_GPU);
  free(Qi_GPU);
  free(Qr_CPU);
  free(Qi_CPU);

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

 __global__ void __generated_kernel_region_0(DATA_TYPE * A,DATA_TYPE * A_GPU,int  N,int  fail){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < N)
{
    if (__accelerator_percentDiff(A [i], A_GPU [i]) > ERROR_THRESHOLD) {
      fail++;
    }
  }

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_1(DATA_TYPE * B,DATA_TYPE * B_GPU,int  N,int  fail){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < N)
{
    if (__accelerator_percentDiff(B [i], B_GPU [i]) > ERROR_THRESHOLD) {
      fail++;
    }
  }

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_2(float * phiMag,struct kValues * kVals,float * kz,float * kx,float * ky,int  numK){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  k;
{
{
{
 k=0+(__kernel_getuid_x);
if( k < numK)
{
    kVals [k].Kx = kx [k];
    kVals [k].Ky = ky [k];
    kVals [k].Kz = kz [k];
    kVals [k].PhiMag = phiMag [k];
  }

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_3(float * phiMag,struct kValues * kVals,float * kz,float * kx,float * ky,int  numK){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  k;
{
{
{
 k=0+(__kernel_getuid_x);
if( k < numK)
{
    kVals [k].Kx = kx [k];
    kVals [k].Ky = ky [k];
    kVals [k].Kz = kz [k];
    kVals [k].PhiMag = phiMag [k];
  }

}
}
}
//append writeback of scalar variables
}

