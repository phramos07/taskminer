/*
   This program performs matrix sum on the CPU with 
   dynamically allocated matrices.
    
    Author: Gleison Souza Diniz Mendonça 
    Date: 04-01-2015
    version 2.0
    
    Run:
    gcc -O3 mat-sum_cpu.c -o mat
    ./mat matrix-size
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

FILE *fil;
FILE *out;

// Initialize matrices.
void init(float* a, float* b, float* c, int s, int l) {
  int i = 0;
	long long int AI1[3];
	AI1[0] = l > 0;
	AI1[1] = (AI1[0] ? l : 0);
	AI1[2] = 4 * AI1[1];
	acc_create((void*) a, AI1[2]);
	acc_copyin((void*) a, AI1[2]);
	acc_create((void*) b, AI1[2]);
	acc_copyin((void*) b, AI1[2]);
	acc_create((void*) c, AI1[2]);
	acc_copyin((void*) c, AI1[2]);
#pragma acc kernels present(a[0:AI1[2]], b[0:AI1[2], c[0:AI1[2]])
#pragma acc loop independent
  for (i = 0; i < l; ++i)
	{
			a[i] = (float) (i % s);
			b[i] = (float) (i % s);
			c[i] = 0.0f;
	}
acc_copyout_and_keep((void*) a, AI1[2]);
acc_copyout_and_keep((void*) b, AI1[2]);
acc_copyout_and_keep((void*) c, AI1[2]);
}

// Print the result matrix.
void print(float* c, int s) {
	int i;
  for (i = 0; i < s; ++i) 
	{
			fprintf(out,"%f ", c[i]);
		fprintf(out,"\n");
	}
}

/// matrix sum algorithm CPU
/// s = size of matrix
void sum_CPU(int s, int l) 
{
  float* vectorA;
  vectorA = (float*) malloc(sizeof(float) * l);
  float b[l];
  float c[l];

  init(vectorA, b, c, s, l);
	
	int i;
	float start, finish, elapsed;
	start = (float) clock() / (CLOCKS_PER_SEC * 1000);
	long long int AI1[3];
	AI1[0] = l > 0;
	AI1[1] = (AI1[0] ? l : 0);
	AI1[2] = 4 * AI1[1];
	acc_create((void*) vectorA, AI1[2]);
	acc_copyin((void*) vectorA, AI1[2]);
	acc_create((void*) b, AI1[2]);
	acc_copyin((void*) b, AI1[2]);
	acc_create((void*) c, AI1[2]);
	acc_copyin((void*) c, AI1[2]);
#pragma acc kernels present(vectorA, b, c)
#pragma acc loop independent
  for (i = 0; i < l; ++i)
	{
			c[i] = vectorA[i] + b[i]; 
	}
	finish = (float) clock() / (CLOCKS_PER_SEC * 1000);
	elapsed = finish - start;
	fprintf(fil,"%.6lf,",elapsed);
	
	print(c, s);
	
acc_copyout_and_keep((void*) vectorA, AI1[2]);
acc_copyout_and_keep((void*) b, AI1[2]);
acc_copyout_and_keep((void*) c, AI1[2]);
}

int main(int argc, char *argv[]) 
{
	if(argc!=2) 
	{
		return 1;
	}
  int SIZE;
  SIZE = atoi(argv[1]);
    
	fil = fopen("time_cpu.csv","a");
	out = fopen("result_cpu.txt","a");

	fprintf(fil,"SIZE,matrix sum cpu,\n");

	
	fprintf(fil,"%d,",SIZE);
	sum_CPU(SIZE, SIZE * SIZE);
	fprintf(fil,"\n");	  

	fclose(fil);
	fclose(out);
	return 0;
}


