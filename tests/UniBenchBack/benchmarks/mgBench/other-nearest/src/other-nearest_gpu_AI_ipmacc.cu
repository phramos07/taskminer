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

typedef struct point {
  int x;
  int y;
}point;

typedef struct sel_points {
  int position;
  float value;
}sel_points;

#define SIZE 1000
#define points 250
#define var SIZE / points
#define default_v 100000.00
#define GPU_DEVICE 0
#define ERROR_THRESHOLD 0.01

 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_0(int  s,point * vector);
 
 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_1(sel_points * selected,int  s);
 
void init(int s, point *vector, sel_points *selected)
{
  int i, j;
  long long int AI1 [8];
  AI1 [0] = s + -1;
  AI1 [1] = 8 * AI1 [0];
  AI1 [2] = 4 + AI1 [1];
  AI1 [3] = AI1 [2] > AI1 [1];
  AI1 [4] = (AI1 [3] ? AI1 [2] : AI1 [1]);
  AI1 [5] = AI1 [4] / 8;
  AI1 [6] = (AI1 [5] > 0);
  AI1 [7] = (AI1 [6] ? AI1 [5] : 0);
    

	ipmacc_prompt((char*)"IPMACC: memory allocation vector\n");
acc_present_or_create((void*)vector,(AI1[7]+0)*sizeof(point ));
	ipmacc_prompt((char*)"IPMACC: memory copyin vector\n");
acc_pcopyin((void*)vector,(AI1[7]+0)*sizeof(point ));


{


    


/* kernel call statement [0, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 0 > gridDim: %d\tblockDim: %d\n",(((abs((int)((s))-(0+0)))/(1)))/256+(((((abs((int)((s))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_0<<<(((abs((int)((s))-(0+0)))/(1)))/256+(((((abs((int)((s))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
s,
(point *)acc_deviceptr((void*)vector));
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
	ipmacc_prompt((char*)"IPMACC: memory copyout vector\n");
acc_copyout_and_keep((void*)vector,(AI1[7]+0)*sizeof(point ));



  long long int AI2 [11];
  AI2 [0] = s + -1;
  AI2 [1] = s * AI2 [0];
  AI2 [2] = AI2 [1] + AI2 [0];
  AI2 [3] = 2 * AI2 [2];
  AI2 [4] = AI2 [3] * 4;
  AI2 [5] = AI2 [2] * 8;
  AI2 [6] = AI2 [4] > AI2 [5];
  AI2 [7] = (AI2 [6] ? AI2 [4] : AI2 [5]);
  AI2 [8] = AI2 [7] / 8;
  AI2 [9] = (AI2 [8] > 0);
  AI2 [10] = (AI2 [9] ? AI2 [8] : 0);
    

	ipmacc_prompt((char*)"IPMACC: memory allocation selected\n");
acc_present_or_create((void*)selected,(AI2[10]+0)*sizeof(sel_points ));
	ipmacc_prompt((char*)"IPMACC: memory copyin selected\n");
acc_pcopyin((void*)selected,(AI2[10]+0)*sizeof(sel_points ));


{


    


/* kernel call statement [1, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 1 > gridDim: %d\tblockDim: %d\n",(((abs((int)((s))-(0+0)))/(1)))/256+(((((abs((int)((s))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_1<<<(((abs((int)((s))-(0+0)))/(1)))/256+(((((abs((int)((s))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(sel_points *)acc_deviceptr((void*)selected),
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



}
	ipmacc_prompt((char*)"IPMACC: memory copyout selected\n");
acc_copyout_and_keep((void*)selected,(AI2[10]+0)*sizeof(sel_points ));



}

 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_2(int  s,point * vector,sel_points * selected);
 
 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_3(sel_points * selected,int  s);
 
void k_nearest_gpu(int s, point *vector, sel_points *selected)
{
  int i, j, m, q;
  q = s * s;

    #pragma omp target device (GPU_DEVICE)
    #pragma omp target map(to: vector[0: s]) map(tofrom: selected[0:q])
  {
  #pragma omp parallel for collapse(2)
    long long int AI1 [52];
    AI1 [0] = s + -1;
    AI1 [1] = 8 * AI1 [0];
    AI1 [2] = 12 + AI1 [1];
    AI1 [3] = s + -2;
    AI1 [4] = -1 * AI1 [0];
    AI1 [5] = AI1 [3] + AI1 [4];
    AI1 [6] = 8 * AI1 [5];
    AI1 [7] = AI1 [2] + AI1 [6];
    AI1 [8] = 4 + AI1 [1];
    AI1 [9] = 8 + AI1 [1];
    AI1 [10] = AI1 [9] + AI1 [6];
    AI1 [11] = AI1 [10] > AI1 [1];
    AI1 [12] = (AI1 [11] ? AI1 [10] : AI1 [1]);
    AI1 [13] = AI1 [8] > AI1 [12];
    AI1 [14] = (AI1 [13] ? AI1 [8] : AI1 [12]);
    AI1 [15] = AI1 [7] > AI1 [14];
    AI1 [16] = (AI1 [15] ? AI1 [7] : AI1 [14]);
    AI1 [17] = AI1 [16] / 8;
    AI1 [18] = (AI1 [17] > 0);
    AI1 [19] = (AI1 [18] ? AI1 [17] : 0);
    AI1 [20] = s * 8;
    AI1 [21] = 2 * s;
    AI1 [22] = AI1 [21] * 4;
    AI1 [23] = AI1 [22] < 8;
    AI1 [24] = (AI1 [23] ? AI1 [22] : 8);
    AI1 [25] = AI1 [20] < AI1 [24];
    AI1 [26] = (AI1 [25] ? AI1 [20] : AI1 [24]);
    AI1 [27] = AI1 [26] / 8;
    AI1 [28] = (AI1 [27] > 0);
    AI1 [29] = (AI1 [28] ? AI1 [27] : 0);
    AI1 [30] = s + 1;
    AI1 [31] = AI1 [30] * AI1 [0];
    AI1 [32] = s + AI1 [31];
    AI1 [33] = s * AI1 [5];
    AI1 [34] = AI1 [32] + AI1 [33];
    AI1 [35] = AI1 [34] * 8;
    AI1 [36] = 2 * AI1 [34];
    AI1 [37] = AI1 [36] * 4;
    AI1 [38] = 1 + AI1 [31];
    AI1 [39] = AI1 [38] + AI1 [5];
    AI1 [40] = AI1 [39] * 8;
    AI1 [41] = 2 * AI1 [39];
    AI1 [42] = AI1 [41] * 4;
    AI1 [43] = AI1 [40] > AI1 [42];
    AI1 [44] = (AI1 [43] ? AI1 [40] : AI1 [42]);
    AI1 [45] = AI1 [37] > AI1 [44];
    AI1 [46] = (AI1 [45] ? AI1 [37] : AI1 [44]);
    AI1 [47] = AI1 [35] > AI1 [46];
    AI1 [48] = (AI1 [47] ? AI1 [35] : AI1 [46]);
    AI1 [49] = AI1 [48] / 8;
    AI1 [50] = (AI1 [49] > 0);
    AI1 [51] = (AI1 [50] ? AI1 [49] : 0);
        

	ipmacc_prompt((char*)"IPMACC: memory allocation vector\n");
acc_present_or_create((void*)vector,(AI1[19]+0)*sizeof(point ));
ipmacc_prompt((char*)"IPMACC: memory allocation selected\n");
acc_present_or_create((void*)selected,(AI1[51]+AI1[29])*sizeof(sel_points ));
	ipmacc_prompt((char*)"IPMACC: memory copyin vector\n");
acc_pcopyin((void*)vector,(AI1[19]+0)*sizeof(point ));
ipmacc_prompt((char*)"IPMACC: memory copyin selected\n");
acc_pcopyin((void*)selected,(AI1[51]+AI1[29])*sizeof(sel_points ));


{


        


/* kernel call statement [2, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 2 > gridDim: %d\tblockDim: %d\n",(((abs((int)((s))-(0+0)))/(1)))/256+(((((abs((int)((s))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_2<<<(((abs((int)((s))-(0+0)))/(1)))/256+(((((abs((int)((s))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
s,
(point *)acc_deviceptr((void*)vector),
(sel_points *)acc_deviceptr((void*)selected));
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
	ipmacc_prompt((char*)"IPMACC: memory copyout vector\n");
acc_copyout_and_keep((void*)vector,(AI1[19]+0)*sizeof(point ));
ipmacc_prompt((char*)"IPMACC: memory copyout selected\n");
acc_copyout_and_keep((void*)selected,(AI1[51]+AI1[29])*sizeof(sel_points ));




    
    
  #pragma omp parallel for collapse(1)
    long long int AI2 [18];
    AI2 [0] = s + -1;
    AI2 [1] = s * AI2 [0];
    AI2 [2] = 1 + AI2 [1];
    AI2 [3] = AI2 [2] + AI2 [0];
    AI2 [4] = s + -2;
    AI2 [5] = -1 * AI2 [0];
    AI2 [6] = AI2 [4] + AI2 [5];
    AI2 [7] = AI2 [3] + AI2 [6];
    AI2 [8] = 2 * AI2 [7];
    AI2 [9] = AI2 [8] * 4;
    AI2 [10] = AI2 [1] + AI2 [0];
    AI2 [11] = 2 * AI2 [10];
    AI2 [12] = AI2 [11] * 4;
    AI2 [13] = AI2 [9] > AI2 [12];
    AI2 [14] = (AI2 [13] ? AI2 [9] : AI2 [12]);
    AI2 [15] = AI2 [14] / 8;
    AI2 [16] = (AI2 [15] > 0);
    AI2 [17] = (AI2 [16] ? AI2 [15] : 0);
    

	ipmacc_prompt((char*)"IPMACC: memory allocation selected\n");
acc_present_or_create((void*)selected,(AI2[17]+0)*sizeof(sel_points ));
	ipmacc_prompt((char*)"IPMACC: memory copyin selected\n");
acc_pcopyin((void*)selected,(AI2[17]+0)*sizeof(sel_points ));


{


    


/* kernel call statement [3, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 3 > gridDim: %d\tblockDim: %d\n",(((abs((int)((s))-(0+0)))/(1)))/256+(((((abs((int)((s))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_3<<<(((abs((int)((s))-(0+0)))/(1)))/256+(((((abs((int)((s))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(sel_points *)acc_deviceptr((void*)selected),
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



}
	ipmacc_prompt((char*)"IPMACC: memory copyout selected\n");
acc_copyout_and_keep((void*)selected,(AI2[17]+0)*sizeof(sel_points ));



  }
}

 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_4(int  s,point * vector,sel_points * selected);
 
void k_nearest_cpu(int s, point *vector, sel_points *selected)
{
  int i, j;
  long long int AI1 [52];
  AI1 [0] = s + -1;
  AI1 [1] = 8 * AI1 [0];
  AI1 [2] = 12 + AI1 [1];
  AI1 [3] = s + -2;
  AI1 [4] = -1 * AI1 [0];
  AI1 [5] = AI1 [3] + AI1 [4];
  AI1 [6] = 8 * AI1 [5];
  AI1 [7] = AI1 [2] + AI1 [6];
  AI1 [8] = 4 + AI1 [1];
  AI1 [9] = 8 + AI1 [1];
  AI1 [10] = AI1 [9] + AI1 [6];
  AI1 [11] = AI1 [10] > AI1 [1];
  AI1 [12] = (AI1 [11] ? AI1 [10] : AI1 [1]);
  AI1 [13] = AI1 [8] > AI1 [12];
  AI1 [14] = (AI1 [13] ? AI1 [8] : AI1 [12]);
  AI1 [15] = AI1 [7] > AI1 [14];
  AI1 [16] = (AI1 [15] ? AI1 [7] : AI1 [14]);
  AI1 [17] = AI1 [16] / 8;
  AI1 [18] = (AI1 [17] > 0);
  AI1 [19] = (AI1 [18] ? AI1 [17] : 0);
  AI1 [20] = s * 8;
  AI1 [21] = 2 * s;
  AI1 [22] = AI1 [21] * 4;
  AI1 [23] = AI1 [22] < 8;
  AI1 [24] = (AI1 [23] ? AI1 [22] : 8);
  AI1 [25] = AI1 [20] < AI1 [24];
  AI1 [26] = (AI1 [25] ? AI1 [20] : AI1 [24]);
  AI1 [27] = AI1 [26] / 8;
  AI1 [28] = (AI1 [27] > 0);
  AI1 [29] = (AI1 [28] ? AI1 [27] : 0);
  AI1 [30] = s + 1;
  AI1 [31] = AI1 [30] * AI1 [0];
  AI1 [32] = s + AI1 [31];
  AI1 [33] = s * AI1 [5];
  AI1 [34] = AI1 [32] + AI1 [33];
  AI1 [35] = AI1 [34] * 8;
  AI1 [36] = 2 * AI1 [34];
  AI1 [37] = AI1 [36] * 4;
  AI1 [38] = 1 + AI1 [31];
  AI1 [39] = AI1 [38] + AI1 [5];
  AI1 [40] = AI1 [39] * 8;
  AI1 [41] = 2 * AI1 [39];
  AI1 [42] = AI1 [41] * 4;
  AI1 [43] = AI1 [40] > AI1 [42];
  AI1 [44] = (AI1 [43] ? AI1 [40] : AI1 [42]);
  AI1 [45] = AI1 [37] > AI1 [44];
  AI1 [46] = (AI1 [45] ? AI1 [37] : AI1 [44]);
  AI1 [47] = AI1 [35] > AI1 [46];
  AI1 [48] = (AI1 [47] ? AI1 [35] : AI1 [46]);
  AI1 [49] = AI1 [48] / 8;
  AI1 [50] = (AI1 [49] > 0);
  AI1 [51] = (AI1 [50] ? AI1 [49] : 0);
    

	ipmacc_prompt((char*)"IPMACC: memory allocation vector\n");
acc_present_or_create((void*)vector,(AI1[19]+0)*sizeof(point ));
ipmacc_prompt((char*)"IPMACC: memory allocation selected\n");
acc_present_or_create((void*)selected,(AI1[51]+AI1[29])*sizeof(sel_points ));
	ipmacc_prompt((char*)"IPMACC: memory copyin vector\n");
acc_pcopyin((void*)vector,(AI1[19]+0)*sizeof(point ));
ipmacc_prompt((char*)"IPMACC: memory copyin selected\n");
acc_pcopyin((void*)selected,(AI1[51]+AI1[29])*sizeof(sel_points ));


{


    


/* kernel call statement [4, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 4 > gridDim: %d\tblockDim: %d\n",(((abs((int)((s))-(0+0)))/(1)))/256+(((((abs((int)((s))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_4<<<(((abs((int)((s))-(0+0)))/(1)))/256+(((((abs((int)((s))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
s,
(point *)acc_deviceptr((void*)vector),
(sel_points *)acc_deviceptr((void*)selected));
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
	ipmacc_prompt((char*)"IPMACC: memory copyout vector\n");
acc_copyout_and_keep((void*)vector,(AI1[19]+0)*sizeof(point ));
ipmacc_prompt((char*)"IPMACC: memory copyout selected\n");
acc_copyout_and_keep((void*)selected,(AI1[51]+AI1[29])*sizeof(sel_points ));



}

 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_5(sel_points * selected,int  s);
 
void order_points(int s, point *vector, sel_points *selected)
{
  int i;
  long long int AI1 [18];
  AI1 [0] = s + -1;
  AI1 [1] = s * AI1 [0];
  AI1 [2] = 1 + AI1 [1];
  AI1 [3] = AI1 [2] + AI1 [0];
  AI1 [4] = s + -2;
  AI1 [5] = -1 * AI1 [0];
  AI1 [6] = AI1 [4] + AI1 [5];
  AI1 [7] = AI1 [3] + AI1 [6];
  AI1 [8] = 2 * AI1 [7];
  AI1 [9] = AI1 [8] * 4;
  AI1 [10] = AI1 [1] + AI1 [0];
  AI1 [11] = 2 * AI1 [10];
  AI1 [12] = AI1 [11] * 4;
  AI1 [13] = AI1 [9] > AI1 [12];
  AI1 [14] = (AI1 [13] ? AI1 [9] : AI1 [12]);
  AI1 [15] = AI1 [14] / 8;
  AI1 [16] = (AI1 [15] > 0);
  AI1 [17] = (AI1 [16] ? AI1 [15] : 0);
    

	ipmacc_prompt((char*)"IPMACC: memory allocation selected\n");
acc_present_or_create((void*)selected,(AI1[17]+0)*sizeof(sel_points ));
	ipmacc_prompt((char*)"IPMACC: memory copyin selected\n");
acc_pcopyin((void*)selected,(AI1[17]+0)*sizeof(sel_points ));


{


    


/* kernel call statement [5, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 5 > gridDim: %d\tblockDim: %d\n",(((abs((int)((s))-(0+0)))/(1)))/256+(((((abs((int)((s))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_5<<<(((abs((int)((s))-(0+0)))/(1)))/256+(((((abs((int)((s))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(sel_points *)acc_deviceptr((void*)selected),
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



}
	ipmacc_prompt((char*)"IPMACC: memory copyout selected\n");
acc_copyout_and_keep((void*)selected,(AI1[17]+0)*sizeof(sel_points ));



}

 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_6(sel_points * B,sel_points * B_GPU,int  fail);
 
void compareResults(sel_points* B, sel_points* B_GPU)
{
  int i, j, fail;
  fail = 0;

  
  

	ipmacc_prompt((char*)"IPMACC: memory allocation B\n");
acc_present_or_create((void*)B,(249999+0)*sizeof(sel_points ));
ipmacc_prompt((char*)"IPMACC: memory allocation B_GPU\n");
acc_present_or_create((void*)B_GPU,(249999+0)*sizeof(sel_points ));
	ipmacc_prompt((char*)"IPMACC: memory copyin B\n");
acc_pcopyin((void*)B,(249999+0)*sizeof(sel_points ));
ipmacc_prompt((char*)"IPMACC: memory copyin B_GPU\n");
acc_pcopyin((void*)B_GPU,(249999+0)*sizeof(sel_points ));


{


  


/* kernel call statement [6, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 6 > gridDim: %d\tblockDim: %d\n",(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_6<<<(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(sel_points *)acc_deviceptr((void*)B),
(sel_points *)acc_deviceptr((void*)B_GPU),
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
acc_copyout_and_keep((void*)B,(249999+0)*sizeof(sel_points ));
ipmacc_prompt((char*)"IPMACC: memory copyout B_GPU\n");
acc_copyout_and_keep((void*)B_GPU,(249999+0)*sizeof(sel_points ));



  
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", ERROR_THRESHOLD, fail);
}

int main(int argc, char *argv[])
{
  double t_start, t_end;
  point *vector;
  sel_points *selected_cpu, *selected_gpu;

  vector = (point*)malloc(sizeof(point) * SIZE);
  selected_cpu = (sel_points*)malloc(sizeof(sel_points) * SIZE * SIZE);
  selected_gpu = (sel_points*)malloc(sizeof(sel_points) * SIZE * SIZE);

  int i;

  fprintf(stdout, "<< Nearest >>\n");

  t_start = rtclock();
  for (i = (var - 1); i < SIZE; i += var) {
    init(i, vector, selected_cpu);
    k_nearest_cpu(i, vector, selected_cpu);
    order_points(i, vector, selected_cpu);
  }
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);


  t_start = rtclock();
  for (i = (var - 1); i < SIZE; i += var) {
    init(i, vector, selected_gpu);
    k_nearest_gpu(i, vector, selected_gpu);
  }
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(selected_cpu, selected_gpu);

  free(selected_cpu);
  free(selected_gpu);
  free(vector);
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

 __global__ void __generated_kernel_region_0(int  s,point * vector){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < s)
{
    vector [i].x = i;
    vector [i].y = i * 2;
  }

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_1(sel_points * selected,int  s){
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
for(j = 0; j < s; j++)
{
      selected [i * s + j].position = 0;
      selected [i * s + j].value = default_v;
    }
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_2(int  s,point * vector,sel_points * selected){
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
for(j = i + 1; j < s; j++)
{
        float distance, x, y;
        x = vector [i].x - vector [j].x;
        y = vector [i].y - vector [j].y;
        x = x * x;
        y = y * y;

        distance = x + y;
        distance = sqrt(distance);

        selected [i * s + j].value = distance;
        selected [i * s + j].position = j;

        selected [j * s + i].value = distance;
        selected [j * s + i].position = i;
      }
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_3(sel_points * selected,int  s){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  j;
int  m;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < s)
{
for(j = 0; j < s; j++)
{
for(m = j + 1; m < s; m++)
{
          if (selected [i * s + j].value > selected [i * s + m].value) {
            sel_points aux;
            aux = selected [i * s + j];
            selected [i * s + j] = selected [i * s + m];
            selected [i * s + m] = aux;
          }
        }
}
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_4(int  s,point * vector,sel_points * selected){
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
for(j = i + 1; j < s; j++)
{
      float distance, x, y;
      x = vector [i].x - vector [j].x;
      y = vector [i].y - vector [j].y;
      x = x * x;
      y = y * y;

      distance = x + y;
      distance = sqrt(distance);

      selected [i * s + j].value = distance;
      selected [i * s + j].position = j;

      selected [j * s + i].value = distance;
      selected [j * s + i].position = i;
    }
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_5(sel_points * selected,int  s){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < s)
{
    
    
    int j;
for(j = 0; j < s; j++)
{
      int m;
for(m = j + 1; m < s; m++)
{
        if (selected [i * s + j].value > selected [i * s + m].value) {
          sel_points aux;
          aux = selected [i * s + j];
          selected [i * s + j] = selected [i * s + m];
          selected [i * s + m] = aux;
        }
      }
}
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_6(sel_points * B,sel_points * B_GPU,int  fail){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  j;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < SIZE)
{
for(j = 0; j < SIZE; j++)
{
      
      if (__accelerator_percentDiff(B [i * SIZE + j].value, B_GPU [i * SIZE + j].value) > ERROR_THRESHOLD) {
        fail++;
      }
      
      if (__accelerator_percentDiff(B [i * SIZE + j].position, B_GPU [i * SIZE + j].position) > ERROR_THRESHOLD) {
        fail++;
      }
    }
}

}
}
}
//append writeback of scalar variables
}

