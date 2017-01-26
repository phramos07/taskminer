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
#include <time.h>
#include <math.h>

FILE *fil;
FILE *out;


  __global__ void __generated_kernel_region_0(float * a,float * c,int  s,float * b);
 
void init(int s, float *a, float *b, float *c)
{
  int i, j;
  long long int AI1 [5];
  AI1 [0] = s > 0;
  AI1 [1] = (AI1 [0] ? s : 0);
  AI1 [2] = s * AI1 [1];
  AI1 [3] = AI1 [2] + s;
  AI1 [4] = AI1 [3] * 4;
  

	ipmacc_prompt((char*)"IPMACC: memory allocation b\n");
acc_create((void*)b,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation c\n");
acc_create((void*)c,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation a\n");
acc_create((void*)a,(AI1[4]+0)*sizeof(float ));
	ipmacc_prompt((char*)"IPMACC: memory copyin b\n");
acc_copyin((void*)b,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin c\n");
acc_copyin((void*)c,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin a\n");
acc_copyin((void*)a,(AI1[4]+0)*sizeof(float ));


{


  


/* kernel call statement [0, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 0 > gridDim: %d\tblockDim: %d\n",(((abs((int)((s))-(0+0)))/(1)))/256+(((((abs((int)((s))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_0<<<(((abs((int)((s))-(0+0)))/(1)))/256+(((((abs((int)((s))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(float *)acc_deviceptr((void*)a),
(float *)acc_deviceptr((void*)c),
s,
(float *)acc_deviceptr((void*)b));
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
	ipmacc_prompt((char*)"IPMACC: memory copyout b\n");
acc_copyout_and_keep((void*)b,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout c\n");
acc_copyout_and_keep((void*)c,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout a\n");
acc_copyout_and_keep((void*)a,(AI1[4]+0)*sizeof(float ));



}


void print(int s, float *c)
{
  int i, j;
  for (i = 0; i < s; ++i) {
    for (j = 0; j < s; ++j) {
      fprintf(out, "%f ", c [i * s + j]);
    }
    fprintf(out, "\n");
  }
}



  __global__ void __generated_kernel_region_1(float * a,float * b,float * c,int  s,float  sum);
 
void mul_CPU(int s, float *a, float *b, float *c)
{
  init(s, a, b, c);

  int i, j, k;
  float sum = 0.0;
  float start, finish, elapsed;
  start = (float)clock() / (CLOCKS_PER_SEC * 1000);
  long long int AI1 [8];
  AI1 [0] = s > 0;
  AI1 [1] = (AI1 [0] ? s : 0);
  AI1 [2] = s * AI1 [1];
  AI1 [3] = AI1 [2] + s;
  AI1 [4] = AI1 [3] * 4;
  AI1 [5] = s * s;
  AI1 [6] = s + AI1 [5];
  AI1 [7] = AI1 [6] * 4;
  

	ipmacc_prompt((char*)"IPMACC: memory allocation c\n");
acc_create((void*)c,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation b\n");
acc_create((void*)b,(AI1[7]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory allocation a\n");
acc_create((void*)a,(AI1[4]+0)*sizeof(float ));
	ipmacc_prompt((char*)"IPMACC: memory copyin c\n");
acc_copyin((void*)c,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin b\n");
acc_copyin((void*)b,(AI1[7]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyin a\n");
acc_copyin((void*)a,(AI1[4]+0)*sizeof(float ));


{


  


/* kernel call statement [1, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 1 > gridDim: %d\tblockDim: %d\n",(((abs((int)((s))-(0+0)))/(1)))/256+(((((abs((int)((s))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_1<<<(((abs((int)((s))-(0+0)))/(1)))/256+(((((abs((int)((s))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(float *)acc_deviceptr((void*)a),
(float *)acc_deviceptr((void*)b),
(float *)acc_deviceptr((void*)c),
s,
sum);
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
	ipmacc_prompt((char*)"IPMACC: memory copyout c\n");
acc_copyout_and_keep((void*)c,(AI1[4]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout b\n");
acc_copyout_and_keep((void*)b,(AI1[7]+0)*sizeof(float ));
ipmacc_prompt((char*)"IPMACC: memory copyout a\n");
acc_copyout_and_keep((void*)a,(AI1[4]+0)*sizeof(float ));



  finish = (float)clock() / (CLOCKS_PER_SEC * 1000);
  elapsed = finish - start;
  fprintf(fil, "%.10lf,", elapsed);

  
  free(a);
  free(b);
  free(c);
}

int main(int argc, char *argv[])
{
  if (argc != 2) {
    return 1;
  }

  int SIZE = atoi(argv [1]);

  fil = fopen("time_cpu.csv", "w+");
  out = fopen("result_cpu.txt", "w+");

  fprintf(fil, "SIZE,matrix multiplication CPU,\n");

  fprintf(fil, "%d,", SIZE);

  float *a, *b, *c;
  a = (float*)malloc(sizeof(float) * SIZE * SIZE);
  b = (float*)malloc(sizeof(float) * SIZE * SIZE);
  c = (float*)malloc(sizeof(float) * SIZE * SIZE);
  mul_CPU(SIZE, a, b, c);
  fprintf(fil, "\n");

  fclose(fil);
  fclose(out);
  return 0;
}



 __global__ void __generated_kernel_region_0(float * a,float * c,int  s,float * b){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  j;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < s)
{
for(j = 0; j < s; ++j)
{
      a [i * s + j] = (float)i + j % 100;
      b [i * s + j] = (float)i + j % 100;
      c [i * s + j] = 0.0f;
    }
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_1(float * a,float * b,float * c,int  s,float  sum){
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
if( i < s)
{
for(j = 0; j < s; ++j)
{
      sum = 0.0;
for(k = 0; k < s; ++k)
{
        sum = sum + a [i * s + k] * b [k * s + j];
      }
c [i * s + j] = sum;
    }
}

}
}
}
//append writeback of scalar variables
}

