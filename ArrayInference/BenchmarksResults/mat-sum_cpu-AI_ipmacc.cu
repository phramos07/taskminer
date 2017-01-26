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


  __global__ void __generated_kernel_region_0(float * a,float * b,float * c,int  l,int  s);
 
void init(float* a, float* b, float* c, int s, int l)
{
  int i = 0;
  long long int AI1 [3];
  AI1 [0] = l > 0;
  AI1 [1] = (AI1 [0] ? l : 0);
  AI1 [2] = 4 * AI1 [1];
  acc_create((void*)a, AI1 [2]);
  acc_copyin((void*)a, AI1 [2]);
  acc_create((void*)b, AI1 [2]);
  acc_copyin((void*)b, AI1 [2]);
  acc_create((void*)c, AI1 [2]);
  acc_copyin((void*)c, AI1 [2]);


		ipmacc_prompt((char*)"IPMACC: memory getting device pointer for a\n");
acc_present((void*)a);
ipmacc_prompt((char*)"IPMACC: memory getting device pointer for b\n");
acc_present((void*)b);
ipmacc_prompt((char*)"IPMACC: memory getting device pointer for c\n");
acc_present((void*)c);

/* kernel call statement [0]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 0 > gridDim: %d\tblockDim: %d\n",(((abs((int)((l))-(0+0)))/(1)))/256+(((((abs((int)((l))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_0<<<(((abs((int)((l))-(0+0)))/(1)))/256+(((((abs((int)((l))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(float *)acc_deviceptr((void*)a),
(float *)acc_deviceptr((void*)b),
(float *)acc_deviceptr((void*)c),
l,
s);
}
/* kernel call statement*/
	if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Synchronizing the region with host\n");
{
cudaError err=cudaDeviceSynchronize();
if(err!=cudaSuccess){
printf("Kernel Launch Error! error code (%d)\n",err);
assert(0&&"Launch Failure!\n");}
}



  acc_copyout_and_keep((void*)a, AI1 [2]);
  acc_copyout_and_keep((void*)b, AI1 [2]);
  acc_copyout_and_keep((void*)c, AI1 [2]);
}


void print(float* c, int s)
{
  int i;
  for (i = 0; i < s; ++i) {
    fprintf(out, "%f ", c [i]);
    fprintf(out, "\n");
  }
}



void sum_CPU(int s, int l)
{
  float* vectorA;
  vectorA = (float*)malloc(sizeof(float) * l);
  float b [l];
  float c [l];

  init(vectorA, b, c, s, l);

  int i;
  float start, finish, elapsed;
  start = (float)clock() / (CLOCKS_PER_SEC * 1000);
  long long int AI1 [3];
  AI1 [0] = l > 0;
  AI1 [1] = (AI1 [0] ? l : 0);
  AI1 [2] = 4 * AI1 [1];
  acc_create((void*)vectorA, AI1 [2]);
  acc_copyin((void*)vectorA, AI1 [2]);
  acc_create((void*)b, AI1 [2]);
  acc_copyin((void*)b, AI1 [2]);
  acc_create((void*)c, AI1 [2]);
  acc_copyin((void*)c, AI1 [2]);
  
  
  for (i = 0; i < l; ++i) {
    c [i] = vectorA [i] + b [i];
  }
  finish = (float)clock() / (CLOCKS_PER_SEC * 1000);
  elapsed = finish - start;
  fprintf(fil, "%.6lf,", elapsed);

  print(c, s);

  acc_copyout_and_keep((void*)vectorA, AI1 [2]);
  acc_copyout_and_keep((void*)b, AI1 [2]);
  acc_copyout_and_keep((void*)c, AI1 [2]);
}

int main(int argc, char *argv[])
{
  if (argc != 2) {
    return 1;
  }
  int SIZE;
  SIZE = atoi(argv [1]);

  fil = fopen("time_cpu.csv", "a");
  out = fopen("result_cpu.txt", "a");

  fprintf(fil, "SIZE,matrix sum cpu,\n");


  fprintf(fil, "%d,", SIZE);
  sum_CPU(SIZE, SIZE * SIZE);
  fprintf(fil, "\n");

  fclose(fil);
  fclose(out);
  return 0;
}



 __global__ void __generated_kernel_region_0(float * a,float * b,float * c,int  l,int  s){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < l)
{
    a [i] = (float)(i % s);
    b [i] = (float)(i % s);
    c [i] = 0.0f;
  }

}
}
}
//append writeback of scalar variables
}

