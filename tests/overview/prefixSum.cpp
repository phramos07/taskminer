#include <iostream>
#include <omp.h>
#define N 1000000000


int main() {
	int* x = (int*) malloc(N*sizeof(int));

	for (long unsigned i=0; i<N; i++)
		x[i] = i%10;
	
	#pragma omp parallel
	#pragma omp single
	for (long unsigned j=1; j<=N-1; j++) {
    
    #pragma omp task depend(in:x[j-1], inout:x[j])
    {
    	x[j] = x[j] + x[j-1];    	
    }
	}

	// for (int i=0;i<N;i++)
	// 	std::cout<<x[i]<< "  ";

	std::cout<<x[N-1];

	return 0;
}

