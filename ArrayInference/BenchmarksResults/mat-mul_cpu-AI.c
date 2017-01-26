/*
   This program performs matrix multiplication on the CPU with 
   dynamically allocated matrices.
    
    Author: Gleison Souza Diniz Mendonça 
    Date: 04-01-2015
    version 2.0
    
    Run:
    gcc -O3 mat-mul_cpu.c -o mat
    ./mat matrix-size
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

FILE *fil;
FILE *out;

// Initialize matrices.
void init (int s, float *a, float *b, float *c) 
{
	int i, j;
	long long int AI1[6];
	AI1[0] = s > 0;
	AI1[1] = (AI1[0] ? s : 0);
	AI1[2] = s * AI1[1];
	AI1[3] = AI1[2] + s;
	AI1[4] = AI1[3] * 4;
	AI1[5] = AI1[4] / 4;
	#pragma acc data copy(b[0:AI1[5]],c[0:AI1[5]],a[0:AI1[5]])
	#pragma acc kernels 
	#pragma acc loop independent
	for (i = 0; i < s; ++i)
	{
		for (j = 0; j < s; ++j)
		{
			a[i * s + j] = (float)i + j % 100;
			b[i * s + j] = (float)i + j % 100;
			c[i * s + j] = 0.0f;
		}
	}
}

// Print the result matrix.
void print (int s, float *c) {
	int i, j;
	for (i = 0; i < s; ++i) 
	{
		for (j = 0; j < s; ++j)
		{
			fprintf(out,"%f ", c[i * s + j]);
		}
		fprintf(out,"\n");
	}
}

/// matrix multiplication algorithm CPU
/// s = size of matrix
void mul_CPU (int s, float *a, float *b, float *c) {
  init(s, a, b, c);

	int i,j,k;
	float sum = 0.0;
	float start, finish, elapsed;
	start = (float) clock() / (CLOCKS_PER_SEC * 1000);
	long long int AI1[10];
	AI1[0] = s > 0;
	AI1[1] = (AI1[0] ? s : 0);
	AI1[2] = s * AI1[1];
	AI1[3] = AI1[2] + s;
	AI1[4] = AI1[3] * 4;
	AI1[5] = AI1[4] / 4;
	AI1[6] = s * s;
	AI1[7] = s + AI1[6];
	AI1[8] = AI1[7] * 4;
	AI1[9] = AI1[8] / 4;
	#pragma acc data copy(c[0:AI1[5]],b[0:AI1[9]],a[0:AI1[5]])
	#pragma acc kernels
	#pragma acc loop independent
	for (i = 0; i < s; ++i) 
	{
		for (j = 0; j < s; ++j) 
		{
			sum = 0.0;
			for (k = 0; k < s; ++k) 
			{
				sum = sum + a[i * s + k] * b[k * s + j];
			}
			c[i * s + j] = sum;
		}
	}
	finish = (float) clock() / (CLOCKS_PER_SEC * 1000);
	elapsed = finish - start;
	fprintf(fil,"%.10lf,",elapsed);
	
	//print(s, c);	
	free(a);
  free(b);
  free(c);
}

int main (int argc, char *argv[]) {
	if(argc!=2)
	    return 1;

	int SIZE = atoi(argv[1]);

	fil = fopen("time_cpu.csv","w+");
	out = fopen("result_cpu.txt","w+");

	fprintf(fil,"SIZE,matrix multiplication CPU,\n");

	fprintf(fil,"%d,",SIZE);
  
  float *a, *b, *c;
  a = (float *) malloc(sizeof(float) * SIZE * SIZE);
  b = (float *) malloc(sizeof(float) * SIZE * SIZE);
  c = (float *) malloc(sizeof(float) * SIZE * SIZE);
  mul_CPU(SIZE, a, b, c);
	fprintf(fil,"\n");	  

	fclose(fil);
	fclose(out);
	return 0;
}


