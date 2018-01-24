
#include <stdlib.h>
#include <stdio.h>


void g();
void f();

//loop expansion
void loop(int* V, int N)
{
	int i, j;
	for (i = 0; i< N; i++)
	{
		for (j = 0; j < N; j++)
		{
			V[i] = j%i | 0x0000FFFF;
		}
		//eventual computations
		V[i] += i*j;
	}
}

//region expansion
void region(int* V, int N)
{
	int i;
	for (i = 0; i< N; i++)
	{
		if (V[i]%i > 3)
		{
			V[i] -= i;
		}
		else
		{
			V[i] -= 2*i;
		}
		V[i] += i*i;
	}
}

//fcallexpansion case 1
void fcall_1(int* V, int N)
{
	int i, j;
	for (i = 0; i< N; i++)
	{
		for (j = 0; j < N; j++)
		{
			g();
		}
		V[i] += i*j + i%j;
	}
}

//fcallexpansion case 2
void fcall_2(int* V, int N)
{
	int i, j;
	for (i = 0; i< N; i++)
	{
		g();
		for (j = 0; j < N; j++)
		{
			f();
			V[i] += i*j;
		}
	}
}

int main(int argc, char const *argv[])
{


}

