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
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include "../../common/mgbenchUtilFunctions.h"

#define SIZE 128
#define SIZE_2 SIZE / 2
#define GPU_DEVICE 0
#define ERROR_THRESHOLD 0.05
#define default_v 100000.00

typedef struct point {
  int x;
  int y;
}point;

typedef struct sel_points {
  int position;
  float value;
}sel_points;


 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_0(point * pivots);
 
 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_1(sel_points * selected_cpu,sel_points * selected_gpu,point * the_points);
 
void init(point *pivots, point *the_points, sel_points *selected_cpu, sel_points *selected_gpu)
{
  int i, j;
    

	ipmacc_prompt((char*)"IPMACC: memory allocation pivots\n");
acc_present_or_create((void*)pivots,(63+0)*sizeof(point ));
	ipmacc_prompt((char*)"IPMACC: memory copyin pivots\n");
acc_pcopyin((void*)pivots,(63+0)*sizeof(point ));


{


    


/* kernel call statement [0, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 0 > gridDim: %d\tblockDim: %d\n",(((abs((int)((SIZE_2))-(0+0)))/(1)))/256+(((((abs((int)((SIZE_2))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_0<<<(((abs((int)((SIZE_2))-(0+0)))/(1)))/256+(((((abs((int)((SIZE_2))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(point *)acc_deviceptr((void*)pivots));
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
	ipmacc_prompt((char*)"IPMACC: memory copyout pivots\n");
acc_copyout_and_keep((void*)pivots,(63+0)*sizeof(point ));




    

	ipmacc_prompt((char*)"IPMACC: memory allocation the_points\n");
acc_present_or_create((void*)the_points,(127+0)*sizeof(point ));
ipmacc_prompt((char*)"IPMACC: memory allocation selected_cpu\n");
acc_present_or_create((void*)selected_cpu,(16383+0)*sizeof(sel_points ));
ipmacc_prompt((char*)"IPMACC: memory allocation selected_gpu\n");
acc_present_or_create((void*)selected_gpu,(16383+0)*sizeof(sel_points ));
	ipmacc_prompt((char*)"IPMACC: memory copyin the_points\n");
acc_pcopyin((void*)the_points,(127+0)*sizeof(point ));
ipmacc_prompt((char*)"IPMACC: memory copyin selected_cpu\n");
acc_pcopyin((void*)selected_cpu,(16383+0)*sizeof(sel_points ));
ipmacc_prompt((char*)"IPMACC: memory copyin selected_gpu\n");
acc_pcopyin((void*)selected_gpu,(16383+0)*sizeof(sel_points ));


{


    


/* kernel call statement [1, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 1 > gridDim: %d\tblockDim: %d\n",(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_1<<<(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(sel_points *)acc_deviceptr((void*)selected_cpu),
(sel_points *)acc_deviceptr((void*)selected_gpu),
(point *)acc_deviceptr((void*)the_points));
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
	ipmacc_prompt((char*)"IPMACC: memory copyout the_points\n");
acc_copyout_and_keep((void*)the_points,(127+0)*sizeof(point ));
ipmacc_prompt((char*)"IPMACC: memory copyout selected_cpu\n");
acc_copyout_and_keep((void*)selected_cpu,(16383+0)*sizeof(sel_points ));
ipmacc_prompt((char*)"IPMACC: memory copyout selected_gpu\n");
acc_copyout_and_keep((void*)selected_gpu,(16383+0)*sizeof(sel_points ));



}

 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_2(point * pivots,sel_points * selected,point * the_points);
 
 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_3(sel_points * selected);
 
void k_nearest_gpu(point *pivots, point *the_points, sel_points *selected)
{
  int i, j, m;

        

	ipmacc_prompt((char*)"IPMACC: memory allocation pivots\n");
acc_present_or_create((void*)pivots,(63+0)*sizeof(point ));
ipmacc_prompt((char*)"IPMACC: memory allocation the_points\n");
acc_present_or_create((void*)the_points,(127+0)*sizeof(point ));
ipmacc_prompt((char*)"IPMACC: memory allocation selected\n");
acc_present_or_create((void*)selected,(8191+0)*sizeof(sel_points ));
	ipmacc_prompt((char*)"IPMACC: memory copyin pivots\n");
acc_pcopyin((void*)pivots,(63+0)*sizeof(point ));
ipmacc_prompt((char*)"IPMACC: memory copyin the_points\n");
acc_pcopyin((void*)the_points,(127+0)*sizeof(point ));
ipmacc_prompt((char*)"IPMACC: memory copyin selected\n");
acc_pcopyin((void*)selected,(8191+0)*sizeof(sel_points ));


{


        


/* kernel call statement [2, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 2 > gridDim: %d\tblockDim: %d\n",(((abs((int)((SIZE_2))-(0+0)))/(1)))/256+(((((abs((int)((SIZE_2))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_2<<<(((abs((int)((SIZE_2))-(0+0)))/(1)))/256+(((((abs((int)((SIZE_2))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(point *)acc_deviceptr((void*)pivots),
(sel_points *)acc_deviceptr((void*)selected),
(point *)acc_deviceptr((void*)the_points));
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
	ipmacc_prompt((char*)"IPMACC: memory copyout pivots\n");
acc_copyout_and_keep((void*)pivots,(63+0)*sizeof(point ));
ipmacc_prompt((char*)"IPMACC: memory copyout the_points\n");
acc_copyout_and_keep((void*)the_points,(127+0)*sizeof(point ));
ipmacc_prompt((char*)"IPMACC: memory copyout selected\n");
acc_copyout_and_keep((void*)selected,(8191+0)*sizeof(sel_points ));




  
  

        

	ipmacc_prompt((char*)"IPMACC: memory allocation selected\n");
acc_present_or_create((void*)selected,(8191+0)*sizeof(sel_points ));
	ipmacc_prompt((char*)"IPMACC: memory copyin selected\n");
acc_pcopyin((void*)selected,(8191+0)*sizeof(sel_points ));


{


        


/* kernel call statement [3, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 3 > gridDim: %d\tblockDim: %d\n",(((abs((int)((SIZE_2))-(0+0)))/(1)))/256+(((((abs((int)((SIZE_2))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_3<<<(((abs((int)((SIZE_2))-(0+0)))/(1)))/256+(((((abs((int)((SIZE_2))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
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
	ipmacc_prompt((char*)"IPMACC: memory copyout selected\n");
acc_copyout_and_keep((void*)selected,(8191+0)*sizeof(sel_points ));



}

 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_4(point * pivots,sel_points * selected,point * the_points);
 
 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_5(sel_points * selected);
 
void k_nearest_cpu(point *pivots, point *the_points, sel_points *selected)
{
  int i, j;
    

	ipmacc_prompt((char*)"IPMACC: memory allocation pivots\n");
acc_present_or_create((void*)pivots,(63+0)*sizeof(point ));
ipmacc_prompt((char*)"IPMACC: memory allocation the_points\n");
acc_present_or_create((void*)the_points,(127+0)*sizeof(point ));
ipmacc_prompt((char*)"IPMACC: memory allocation selected\n");
acc_present_or_create((void*)selected,(8191+0)*sizeof(sel_points ));
	ipmacc_prompt((char*)"IPMACC: memory copyin pivots\n");
acc_pcopyin((void*)pivots,(63+0)*sizeof(point ));
ipmacc_prompt((char*)"IPMACC: memory copyin the_points\n");
acc_pcopyin((void*)the_points,(127+0)*sizeof(point ));
ipmacc_prompt((char*)"IPMACC: memory copyin selected\n");
acc_pcopyin((void*)selected,(8191+0)*sizeof(sel_points ));


{


    


/* kernel call statement [4, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 4 > gridDim: %d\tblockDim: %d\n",(((abs((int)((SIZE_2))-(0+0)))/(1)))/256+(((((abs((int)((SIZE_2))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_4<<<(((abs((int)((SIZE_2))-(0+0)))/(1)))/256+(((((abs((int)((SIZE_2))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(point *)acc_deviceptr((void*)pivots),
(sel_points *)acc_deviceptr((void*)selected),
(point *)acc_deviceptr((void*)the_points));
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
	ipmacc_prompt((char*)"IPMACC: memory copyout pivots\n");
acc_copyout_and_keep((void*)pivots,(63+0)*sizeof(point ));
ipmacc_prompt((char*)"IPMACC: memory copyout the_points\n");
acc_copyout_and_keep((void*)the_points,(127+0)*sizeof(point ));
ipmacc_prompt((char*)"IPMACC: memory copyout selected\n");
acc_copyout_and_keep((void*)selected,(8191+0)*sizeof(sel_points ));




    

	ipmacc_prompt((char*)"IPMACC: memory allocation selected\n");
acc_present_or_create((void*)selected,(8191+0)*sizeof(sel_points ));
	ipmacc_prompt((char*)"IPMACC: memory copyin selected\n");
acc_pcopyin((void*)selected,(8191+0)*sizeof(sel_points ));


{


    


/* kernel call statement [5, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 5 > gridDim: %d\tblockDim: %d\n",(((abs((int)((SIZE_2))-(0+0)))/(1)))/256+(((((abs((int)((SIZE_2))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_5<<<(((abs((int)((SIZE_2))-(0+0)))/(1)))/256+(((((abs((int)((SIZE_2))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
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
	ipmacc_prompt((char*)"IPMACC: memory copyout selected\n");
acc_copyout_and_keep((void*)selected,(8191+0)*sizeof(sel_points ));



}

 __device__ float __accelerator_absVal( float a );
__device__ float __accelerator_percentDiff( double val1 , double val2 );
 __global__ void __generated_kernel_region_6(sel_points * B,sel_points * B_GPU,int  fail);
 
void compareResults(sel_points* B, sel_points* B_GPU)
{
  int i, j, fail;
  fail = 0;

  
  

	ipmacc_prompt((char*)"IPMACC: memory allocation B\n");
acc_present_or_create((void*)B,(16383+0)*sizeof(sel_points ));
ipmacc_prompt((char*)"IPMACC: memory allocation B_GPU\n");
acc_present_or_create((void*)B_GPU,(16383+0)*sizeof(sel_points ));
	ipmacc_prompt((char*)"IPMACC: memory copyin B\n");
acc_pcopyin((void*)B,(16383+0)*sizeof(sel_points ));
ipmacc_prompt((char*)"IPMACC: memory copyin B_GPU\n");
acc_pcopyin((void*)B_GPU,(16383+0)*sizeof(sel_points ));


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
acc_copyout_and_keep((void*)B,(16383+0)*sizeof(sel_points ));
ipmacc_prompt((char*)"IPMACC: memory copyout B_GPU\n");
acc_copyout_and_keep((void*)B_GPU,(16383+0)*sizeof(sel_points ));



  
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", ERROR_THRESHOLD, fail);
}

int main(int argc, char *argv[])
{
  double t_start, t_end;
  point *pivots;
  point *the_points;
  sel_points *selected_cpu, *selected_gpu;

  fprintf(stdout, "<< K-nearest >>\n");

  pivots = (point*)malloc(sizeof(point) * SIZE);
  the_points = (point*)malloc(sizeof(point) * SIZE);
  selected_cpu = (sel_points*)malloc(sizeof(sel_points) * SIZE * SIZE);
  selected_gpu = (sel_points*)malloc(sizeof(sel_points) * SIZE * SIZE);

  init(pivots, the_points, selected_cpu, selected_gpu);

  t_start = rtclock();
  k_nearest_gpu(pivots, the_points, selected_gpu);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  k_nearest_cpu(pivots, the_points, selected_cpu);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(selected_cpu, selected_gpu);

  free(selected_cpu);
  free(selected_gpu);
  free(pivots);
  free(the_points);

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

 __global__ void __generated_kernel_region_0(point * pivots){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < SIZE_2)
{
    pivots [i].x = i * 3;
    pivots [i].y = i * 2;
  }

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_1(sel_points * selected_cpu,sel_points * selected_gpu,point * the_points){
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
    the_points [i].x = i * 3;
    the_points [i].y = i * 2;
for(j = 0; j < SIZE; j++)
{
      selected_cpu [i * SIZE + j].position = 0;
      selected_cpu [i * SIZE + j].value = default_v;
      selected_gpu [i * SIZE + j].position = 0;
      selected_gpu [i * SIZE + j].value = default_v;
    }
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_2(point * pivots,sel_points * selected,point * the_points){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  j;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < SIZE_2)
{
for(j = 0; j < SIZE; j++)
{
      float distance, x, y;
      x = pivots [i].x - the_points [j].x;
      y = pivots [i].y - the_points [j].y;
      x = x * x;
      y = y * y;

      distance = x + y;
      distance = sqrt(distance);

      selected [i * SIZE + j].value = distance;
      selected [i * SIZE + j].position = j;
    }
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_3(sel_points * selected){
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
if( i < SIZE_2)
{
for(j = 0; j < SIZE; j++)
{
for(m = j + 1; m < SIZE; m++)
{
        if (selected [i * SIZE + j].value > selected [i * SIZE + m].value) {
          sel_points aux;
          aux = selected [i * SIZE + j];
          selected [i * SIZE + j] = selected [i * SIZE + m];
          selected [i * SIZE + m] = aux;
        }
      }
}
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_4(point * pivots,sel_points * selected,point * the_points){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  j;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < SIZE_2)
{
for(j = 0; j < SIZE; j++)
{
      float distance, x, y;
      x = pivots [i].x - the_points [j].x;
      y = pivots [i].y - the_points [j].y;
      x = x * x;
      y = y * y;

      distance = x + y;
      distance = sqrt(distance);

      selected [i * SIZE + j].value = distance;
      selected [i * SIZE + j].position = j;
    }
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_5(sel_points * selected){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  j;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < SIZE_2)
{
    
    
    int j;
for(j = 0; j < SIZE; j++)
{
      int m;
for(m = j + 1; m < SIZE; m++)
{
        if (selected [i * SIZE + j].value > selected [i * SIZE + m].value) {
          sel_points aux;
          aux = selected [i * SIZE + j];
          selected [i * SIZE + j] = selected [i * SIZE + m];
          selected [i * SIZE + m] = aux;
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

