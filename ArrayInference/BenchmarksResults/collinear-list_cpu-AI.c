/*
    This program checks the collinearity of points.
    It receives an input a vector with points and returns the mathematical functions that pass these points. It have a list to store answers.
    This program create a csv file with the time execution results for each function(CPU,GPU) in this format: size of vector, cpu with list time, gpu with list time.
    
    Author: Gleison Souza Diniz Mendonça 
    Date: 04-05-2015
    version 2.0
    
    Run:
    folder_ipmacc/ipmacc folder_archive/colinear_v2.c
    ./a.out
*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <string.h>

typedef struct point
{
	int x;
	int y;
} point;

int SIZE;

FILE *fil;
FILE *out;

void generate_points(int size, point *points)
{
	int i;
     long long int AI1[6];
     AI1[0] = size > 0;
     AI1[1] = (AI1[0] ? size : 0);
     AI1[2] = 8 * AI1[1];
     AI1[3] = 4 + AI1[2];
     AI1[4] = AI1[3] > AI1[2];
     AI1[5] = (AI1[4] ? AI1[3] : AI1[2]);
     acc_create((void*) points, AI1[5]);
     acc_copyin((void*) points, AI1[5]);
     for(i=0;i<size;i++)
	{
		points[i].x = (i*777)%11;
		points[i].y = (i*777)%13;
	}
acc_copyout_and_keep((void*) points, AI1[5]);
}

/// colinear list algorithm CPU
/// N = size of vector
int colinear_list_points_CPU(int N)
{
	
	int i,j,k,val;
	val = 0;
  point *points;
	points = (point *) malloc(sizeof(points)*N);
	generate_points(N, points);
	
	float start, finish, elapsed;
	start = (float) clock() / (CLOCKS_PER_SEC * 1000);
	long long int AI1[24];
	AI1[0] = 8 * N;
	AI1[1] = 4 + AI1[0];
	AI1[2] = N > 0;
	AI1[3] = (AI1[2] ? N : 0);
	AI1[4] = 8 * AI1[3];
	AI1[5] = 4 + AI1[4];
	AI1[6] = AI1[5] > AI1[1];
	AI1[7] = (AI1[6] ? AI1[5] : AI1[1]);
	AI1[8] = AI1[0] > AI1[7];
	AI1[9] = (AI1[8] ? AI1[0] : AI1[7]);
	AI1[10] = AI1[4] > AI1[9];
	AI1[11] = (AI1[10] ? AI1[4] : AI1[9]);
	AI1[12] = AI1[0] > AI1[11];
	AI1[13] = (AI1[12] ? AI1[0] : AI1[11]);
	AI1[14] = AI1[4] > AI1[13];
	AI1[15] = (AI1[14] ? AI1[4] : AI1[13]);
	AI1[16] = AI1[5] > AI1[15];
	AI1[17] = (AI1[16] ? AI1[5] : AI1[15]);
	AI1[18] = AI1[4] > AI1[17];
	AI1[19] = (AI1[18] ? AI1[4] : AI1[17]);
	AI1[20] = AI1[1] > AI1[19];
	AI1[21] = (AI1[20] ? AI1[1] : AI1[19]);
	AI1[22] = AI1[0] > AI1[21];
	AI1[23] = (AI1[22] ? AI1[0] : AI1[21]);
	acc_create((void*) points, AI1[23]);
	acc_copyin((void*) points, AI1[23]);
	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++)
		{
			for(k = 0; k < N; k++)
			{
				/// to understand if is colinear points
				int slope_coefficient,linear_coefficient;
				int ret;
				ret = 0;
				slope_coefficient = points[j].y - points[i].y;
				
				if((points[j].x - points[i].x)!=0)
				{
					slope_coefficient = slope_coefficient / (points[j].x - points[i].x);
					linear_coefficient = points[i].y - (points[i].x * slope_coefficient);
					
					if(slope_coefficient!=0
					&&linear_coefficient!=0
					&&points[k].y == (points[k].x * slope_coefficient) + linear_coefficient)
					{

						ret = 1;
					}
				}
				/// to list add
				if(ret==1)
				{
					val = 1;
				}
			}
		}
	}
	finish = (float) clock() / (CLOCKS_PER_SEC * 1000);
	elapsed = finish - start;
	fprintf(fil,"%.10lf,",elapsed);
    free(points);
    acc_copyout_and_keep((void*) points, AI1[23]);
    return val;
}

int main(int argc, char *argv[])
{
	if(argc!=2) 
	{
		return 1;
	}
	SIZE = atoi(argv[1]);
	
	fil = fopen("time_cpu.csv","a+");
	out = fopen("result_cpu.txt","a+");
    
  	fprintf(fil,"SIZE,collinear list CPU,\n");
	fprintf(fil,"%d,",SIZE);
	fprintf(out,"%d\n",colinear_list_points_CPU(SIZE));
	fprintf(fil,"\n");

	fclose(fil);
	fclose(out);
	return 0;
}

