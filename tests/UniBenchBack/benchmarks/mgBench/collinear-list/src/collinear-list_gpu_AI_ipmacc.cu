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
#include <limits.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include "../../common/mgbenchUtilFunctions.h"

#define SIZE 1024
#define GPU_DEVICE 0
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

typedef struct point {
  int x;
  int y;
} point;

point *points;

  __global__ void __generated_kernel_region_0(point * points);
 
void generate_points()
{
  int i;
    

	ipmacc_prompt((char*)"IPMACC: memory allocation points\n");
acc_present_or_create((void*)points,(1023+0)*sizeof(point ));
	ipmacc_prompt((char*)"IPMACC: memory copyin points\n");
acc_pcopyin((void*)points,(1023+0)*sizeof(point ));


{


    


/* kernel call statement [0, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 0 > gridDim: %d\tblockDim: %d\n",(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_0<<<(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(point *)acc_deviceptr((void*)points));
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
	ipmacc_prompt((char*)"IPMACC: memory copyout points\n");
acc_copyout_and_keep((void*)points,(1023+0)*sizeof(point ));



}



  __global__ void __generated_kernel_region_1(int  p,int * parallel_lines);
 
  __global__ void __generated_kernel_region_2(int  p,int * parallel_lines,point * points);
 
int colinear_list_points_GPU()
{
  int i, j, k, p, val;
  val = 0;
  p = 10000;

  int *parallel_lines;
  parallel_lines = (int*)malloc(sizeof(int) * p);
    

	ipmacc_prompt((char*)"IPMACC: memory allocation parallel_lines\n");
acc_present_or_create((void*)parallel_lines,(9999+0)*sizeof(int ));
	ipmacc_prompt((char*)"IPMACC: memory copyin parallel_lines\n");
acc_pcopyin((void*)parallel_lines,(9999+0)*sizeof(int ));


{


    


/* kernel call statement [1, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 1 > gridDim: %d\tblockDim: %d\n",(((abs((int)((p))-(0+0)))/(1)))/256+(((((abs((int)((p))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_1<<<(((abs((int)((p))-(0+0)))/(1)))/256+(((((abs((int)((p))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
p,
(int *)acc_deviceptr((void*)parallel_lines));
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
	ipmacc_prompt((char*)"IPMACC: memory copyout parallel_lines\n");
acc_copyout_and_keep((void*)parallel_lines,(9999+0)*sizeof(int ));




    #pragma omp target device (GPU_DEVICE)
    #pragma omp target map(to: points[0:SIZE]) map(tofrom: parallel_lines[0:p])
  {
        #pragma omp parallel for collapse(3)
  

	ipmacc_prompt((char*)"IPMACC: memory allocation points\n");
acc_present_or_create((void*)points,(1023+0)*sizeof(point ));
ipmacc_prompt((char*)"IPMACC: memory allocation parallel_lines\n");
acc_present_or_create((void*)parallel_lines,(10000+0)*sizeof(int ));
	ipmacc_prompt((char*)"IPMACC: memory copyin points\n");
acc_pcopyin((void*)points,(1023+0)*sizeof(point ));
ipmacc_prompt((char*)"IPMACC: memory copyin parallel_lines\n");
acc_pcopyin((void*)parallel_lines,(10000+0)*sizeof(int ));


{


  


/* kernel call statement [2, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 2 > gridDim: %d\tblockDim: %d\n",(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_2<<<(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
p,
(int *)acc_deviceptr((void*)parallel_lines),
(point *)acc_deviceptr((void*)points));
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
	ipmacc_prompt((char*)"IPMACC: memory copyout points\n");
acc_copyout_and_keep((void*)points,(1023+0)*sizeof(point ));
ipmacc_prompt((char*)"IPMACC: memory copyout parallel_lines\n");
acc_copyout_and_keep((void*)parallel_lines,(10000+0)*sizeof(int ));



  }

  val = 0;
  for (i = 0; i < p; i++) {
    if (parallel_lines [i] == 1) {
      val = 1;
      break;
    }
  }

  free(parallel_lines);

  return val;
}

  __global__ void __generated_kernel_region_3(int  val,point * points);
 
int colinear_list_points_CPU()
{
  int i, j, k, val;
  val = 0;

    

	ipmacc_prompt((char*)"IPMACC: memory allocation points\n");
acc_present_or_create((void*)points,(1023+0)*sizeof(point ));
	ipmacc_prompt((char*)"IPMACC: memory copyin points\n");
acc_pcopyin((void*)points,(1023+0)*sizeof(point ));


{


    


/* kernel call statement [3, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 3 > gridDim: %d\tblockDim: %d\n",(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_3<<<(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
val,
(point *)acc_deviceptr((void*)points));
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
	ipmacc_prompt((char*)"IPMACC: memory copyout points\n");
acc_copyout_and_keep((void*)points,(1023+0)*sizeof(point ));




  return val;
}

void compareResults(int A, int A_outputFromGpu)
{
  int i, j, fail;
  fail = 0;
  if (A != A_outputFromGpu) {
    fail = 1;
  }

  
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

int main(int argc, char *argv[])
{
  double t_start, t_end;
  int result_CPU, result_GPU;

  fprintf(stdout, "<< Collinear List >>\n");

  points = (point*)malloc(sizeof(points) * SIZE);
  generate_points();

  t_start = rtclock();
  result_GPU = colinear_list_points_GPU();
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);


  t_start = rtclock();
  result_CPU = colinear_list_points_CPU();
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(result_GPU, result_CPU);

  free(points);

  return 0;
}



 __global__ void __generated_kernel_region_0(point * points){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < SIZE)
{
    points [i].x = (i * 777) % 11;
    points [i].y = (i * 777) % 13;
  }

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_1(int  p,int * parallel_lines){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < p)
{
    parallel_lines [i] = 0;
  }

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_2(int  p,int * parallel_lines,point * points){
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
if( i < SIZE)
{
for(j = 0; j < SIZE; j++)
{
for(k = 0; k < SIZE; k++)
{
          
          int slope_coefficient, linear_coefficient;
          int ret;
          ret = 0;
          slope_coefficient = points [j].y - points [i].y;
          if ((points [j].x - points [i].x) != 0) {
            slope_coefficient = slope_coefficient / (points [j].x - points [i].x);
            linear_coefficient = points [i].y - (points [i].x * slope_coefficient);
            if (slope_coefficient != 0 && linear_coefficient != 0 &&
                points [k].y == (points [k].x * slope_coefficient) + linear_coefficient) {
              ret = 1;
            }
          }
          if (ret == 1) {
            parallel_lines [(i % p)] = 1;
          }
        }
}
}

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_3(int  val,point * points){
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
if( i < SIZE)
{
for(j = 0; j < SIZE; j++)
{
for(k = 0; k < SIZE; k++)
{
        
        int slope_coefficient, linear_coefficient;
        int ret;
        ret = 0;
        slope_coefficient = points [j].y - points [i].y;

        if ((points [j].x - points [i].x) != 0) {
          slope_coefficient = slope_coefficient / (points [j].x - points [i].x);
          linear_coefficient = points [i].y - (points [i].x * slope_coefficient);

          if (slope_coefficient != 0 &&
              linear_coefficient != 0 &&
              points [k].y == (points [k].x * slope_coefficient) + linear_coefficient) {
            ret = 1;
          }
        }
        
        if (ret == 1) {
          val = 1;
        }
      }
}
}

}
}
}
//append writeback of scalar variables
}

