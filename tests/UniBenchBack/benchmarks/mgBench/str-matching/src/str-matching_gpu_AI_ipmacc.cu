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
#include <limits.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include "../../common/mgbenchUtilFunctions.h"

#define SIZE 1000
#define SIZE2 500
#define GPU_DEVICE 0
#define PERCENT_DIFF_ERROR_THRESHOLD 0.01


  __global__ void __generated_kernel_region_0(char * frase);
 
  __global__ void __generated_kernel_region_1(char * palavra);
 
void init(char *frase, char *palavra)
{
  int i;
    

	ipmacc_prompt((char*)"IPMACC: memory allocation frase\n");
acc_present_or_create((void*)frase,(999+0)*sizeof(char ));
	ipmacc_prompt((char*)"IPMACC: memory copyin frase\n");
acc_pcopyin((void*)frase,(999+0)*sizeof(char ));


{


    


/* kernel call statement [0, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 0 > gridDim: %d\tblockDim: %d\n",(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_0<<<(((abs((int)((SIZE))-(0+0)))/(1)))/256+(((((abs((int)((SIZE))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(char *)acc_deviceptr((void*)frase));
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
	ipmacc_prompt((char*)"IPMACC: memory copyout frase\n");
acc_copyout_and_keep((void*)frase,(999+0)*sizeof(char ));




  frase [i] = '\0';
    

	ipmacc_prompt((char*)"IPMACC: memory allocation palavra\n");
acc_present_or_create((void*)palavra,(499+0)*sizeof(char ));
	ipmacc_prompt((char*)"IPMACC: memory copyin palavra\n");
acc_pcopyin((void*)palavra,(499+0)*sizeof(char ));


{


    


/* kernel call statement [1, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 1 > gridDim: %d\tblockDim: %d\n",(((abs((int)((SIZE2))-(0+0)))/(1)))/256+(((((abs((int)((SIZE2))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_1<<<(((abs((int)((SIZE2))-(0+0)))/(1)))/256+(((((abs((int)((SIZE2))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(char *)acc_deviceptr((void*)palavra));
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
	ipmacc_prompt((char*)"IPMACC: memory copyout palavra\n");
acc_copyout_and_keep((void*)palavra,(499+0)*sizeof(char ));




  palavra [i] = '\0';
}




  __global__ void __generated_kernel_region_2(int * vector,int  parallel_size);
 
  __global__ void __generated_kernel_region_3(char * frase,int * vector,int  parallel_size,int  diff,char * palavra);
 
  __global__ void __generated_kernel_region_4(int  count,int * vector,int  parallel_size);
 
int string_matching_GPU(char *frase, char *palavra)
{
  int i, diff, j, parallel_size, count = 0;
  diff = SIZE - SIZE2;


  parallel_size = 10000;
  int *vector;
  vector = (int*)malloc(sizeof(int) * parallel_size);

    

	ipmacc_prompt((char*)"IPMACC: memory allocation vector\n");
acc_present_or_create((void*)vector,(9999+0)*sizeof(int ));
	ipmacc_prompt((char*)"IPMACC: memory copyin vector\n");
acc_pcopyin((void*)vector,(9999+0)*sizeof(int ));


{


    


/* kernel call statement [2, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 2 > gridDim: %d\tblockDim: %d\n",(((abs((int)((parallel_size))-(0+0)))/(1)))/256+(((((abs((int)((parallel_size))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_2<<<(((abs((int)((parallel_size))-(0+0)))/(1)))/256+(((((abs((int)((parallel_size))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(int *)acc_deviceptr((void*)vector),
parallel_size);
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
acc_copyout_and_keep((void*)vector,(9999+0)*sizeof(int ));




    #pragma omp target device (GPU_DEVICE)
    #pragma omp target map(to: frase[0:SIZE], palavra[0:SIZE2]) map(tofrom: vector[0:parallel_size])
  {
  #pragma omp parallel for
  

	ipmacc_prompt((char*)"IPMACC: memory allocation palavra\n");
acc_present_or_create((void*)palavra,(499+0)*sizeof(char ));
ipmacc_prompt((char*)"IPMACC: memory allocation frase\n");
acc_present_or_create((void*)frase,(998+0)*sizeof(char ));
ipmacc_prompt((char*)"IPMACC: memory allocation vector\n");
acc_present_or_create((void*)vector,(10000+0)*sizeof(int ));
	ipmacc_prompt((char*)"IPMACC: memory copyin palavra\n");
acc_pcopyin((void*)palavra,(499+0)*sizeof(char ));
ipmacc_prompt((char*)"IPMACC: memory copyin frase\n");
acc_pcopyin((void*)frase,(998+0)*sizeof(char ));
ipmacc_prompt((char*)"IPMACC: memory copyin vector\n");
acc_pcopyin((void*)vector,(10000+0)*sizeof(int ));


{


  


/* kernel call statement [3, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 3 > gridDim: %d\tblockDim: %d\n",(((abs((int)((diff))-(0+0)))/(1)))/256+(((((abs((int)((diff))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_3<<<(((abs((int)((diff))-(0+0)))/(1)))/256+(((((abs((int)((diff))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
(char *)acc_deviceptr((void*)frase),
(int *)acc_deviceptr((void*)vector),
parallel_size,
diff,
(char *)acc_deviceptr((void*)palavra));
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
	ipmacc_prompt((char*)"IPMACC: memory copyout palavra\n");
acc_copyout_and_keep((void*)palavra,(499+0)*sizeof(char ));
ipmacc_prompt((char*)"IPMACC: memory copyout frase\n");
acc_copyout_and_keep((void*)frase,(998+0)*sizeof(char ));
ipmacc_prompt((char*)"IPMACC: memory copyout vector\n");
acc_copyout_and_keep((void*)vector,(10000+0)*sizeof(int ));



  }


    

	ipmacc_prompt((char*)"IPMACC: memory allocation vector\n");
acc_present_or_create((void*)vector,(9999+0)*sizeof(int ));
	ipmacc_prompt((char*)"IPMACC: memory copyin vector\n");
acc_pcopyin((void*)vector,(9999+0)*sizeof(int ));


{


    


/* kernel call statement [4, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 4 > gridDim: %d\tblockDim: %d\n",(((abs((int)((parallel_size))-(0+0)))/(1)))/256+(((((abs((int)((parallel_size))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_4<<<(((abs((int)((parallel_size))-(0+0)))/(1)))/256+(((((abs((int)((parallel_size))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
count,
(int *)acc_deviceptr((void*)vector),
parallel_size);
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
acc_copyout_and_keep((void*)vector,(9999+0)*sizeof(int ));




  return count;
}

  __global__ void __generated_kernel_region_5(int  count,char * frase,int  diff,char * palavra);
 
int string_matching_CPU(char *frase, char *palavra)
{
  int i, j, diff, count;
  diff = SIZE - SIZE2;
  count = 0;

  

	ipmacc_prompt((char*)"IPMACC: memory allocation frase\n");
acc_present_or_create((void*)frase,(998+0)*sizeof(char ));
ipmacc_prompt((char*)"IPMACC: memory allocation palavra\n");
acc_present_or_create((void*)palavra,(499+0)*sizeof(char ));
	ipmacc_prompt((char*)"IPMACC: memory copyin frase\n");
acc_pcopyin((void*)frase,(998+0)*sizeof(char ));
ipmacc_prompt((char*)"IPMACC: memory copyin palavra\n");
acc_pcopyin((void*)palavra,(499+0)*sizeof(char ));


{


  


/* kernel call statement [5, -1]*/
{
if (getenv("IPMACC_VERBOSE")) printf("IPMACC: Launching kernel 5 > gridDim: %d\tblockDim: %d\n",(((abs((int)((diff))-(0+0)))/(1)))/256+(((((abs((int)((diff))-(0+0)))/(1)))%(256))==0?0:1),256);
__generated_kernel_region_5<<<(((abs((int)((diff))-(0+0)))/(1)))/256+(((((abs((int)((diff))-(0+0)))/(1)))%(256))==0?0:1),256>>>(
count,
(char *)acc_deviceptr((void*)frase),
diff,
(char *)acc_deviceptr((void*)palavra));
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
	ipmacc_prompt((char*)"IPMACC: memory copyout frase\n");
acc_copyout_and_keep((void*)frase,(998+0)*sizeof(char ));
ipmacc_prompt((char*)"IPMACC: memory copyout palavra\n");
acc_copyout_and_keep((void*)palavra,(499+0)*sizeof(char ));




  return count;
}

int main(int argc, char *argv[])
{
  double t_start, t_end;
  char *frase;
  char *palavra;

  int count_cpu, count_gpu;

  frase = (char*)malloc(sizeof(char) * (SIZE + 1));
  palavra = (char*)malloc(sizeof(char) * (SIZE2 + 1));

  init(frase, palavra);

  fprintf(stdout, "<< String Matching >>\n");

  t_start = rtclock();
  count_cpu = string_matching_CPU(frase, palavra);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  count_gpu = string_matching_GPU(frase, palavra);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  if (count_cpu == count_gpu) {
    printf("Corrects answers: %d = %d\n", count_cpu, count_gpu);
  } else{
    printf("Error: %d != %d\n", count_cpu, count_gpu);
  }

  free(frase);
  free(palavra);

  return 0;
}



 __global__ void __generated_kernel_region_0(char * frase){
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
    frase [i] = 'a';
  }

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_1(char * palavra){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < SIZE2)
{
    palavra [i] = 'a';
  }

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_2(int * vector,int  parallel_size){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < parallel_size)
{
    vector [i] = 0;
  }

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_3(char * frase,int * vector,int  parallel_size,int  diff,char * palavra){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  j;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < diff)
{
      int v;
      v = 0;
for(j = 0; j < SIZE2; j++)
{
        if (frase [(i + j)] != palavra [j]) {
          v = 1;
        }
      }
if (v == 0) {
        vector [i % parallel_size]++;
      }
    }

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_4(int  count,int * vector,int  parallel_size){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < parallel_size)
{
    count += vector [i];
  }

}
}
}
//append writeback of scalar variables
}

 __global__ void __generated_kernel_region_5(int  count,char * frase,int  diff,char * palavra){
int __kernel_getuid_x=threadIdx.x+blockIdx.x*blockDim.x;
int __kernel_getuid_y=threadIdx.y+blockIdx.y*blockDim.y;
int __kernel_getuid_z=threadIdx.z+blockIdx.z*blockDim.z;
int  i;
int  j;
{
{
{
 i=0+(__kernel_getuid_x);
if( i < diff)
{
    int v;
    v = 0;
for(j = 0; j < SIZE2; j++)
{
      if (frase [(i + j)] != palavra [j]) {
        v = 1;
      }
    }
if (v == 0) {
      count++;
    }
  }

}
}
}
//append writeback of scalar variables
}

