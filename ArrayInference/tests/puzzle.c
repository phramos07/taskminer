// Automatically modified code.
#include "stdlib.h"
#include "stdio.h"

#define ARRAY_SIZE 500000
#define NLOOPS1 5
#define NLOOPS2 200

// RNG implemented localy to avoid library incongruences
#ifdef RAND_MAX
#undef RAND_MAX
#endif
#define RAND_MAX 32767
static unsigned long long int next = 1;
 


//Generated clone.
int GPU__main__createRandomArray__shuffle__rand( void ) {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next / 65536) % RAND_MAX+1;
}



//Generated clone.
int CPU__main__createRandomArray__shuffle__rand( void ) {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next / 65536) % RAND_MAX+1;
}



//Generated clone.
int GPU__main__createRandomArray__randInt__rand( void ) {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next / 65536) % RAND_MAX+1;
}



//Generated clone.
int CPU__main__createRandomArray__randInt__rand( void ) {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next / 65536) % RAND_MAX+1;
}

int rand( void ) {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next / 65536) % RAND_MAX+1;
}
 


//Generated clone.
void GPU__main__srand( unsigned int seed ) {
    next = seed;
}



//Generated clone.
void CPU__main__srand( unsigned int seed ) {
    next = seed;
}

void srand( unsigned int seed ) {
    next = seed;
}
// End of RNG implementation



//Generated clone.
int GPU__main__createRandomArray__randInt(int min, int max) {
    int k, n;
    n = (max - min) + 1;
    k = (int)(n * (CPU__main__createRandomArray__randInt__rand() / (RAND_MAX + 1.0)));
    return (k == n) ? k + min - 1 : k + min;
}



//Generated clone.
int CPU__main__createRandomArray__randInt(int min, int max) {
    int k, n;
    n = (max - min) + 1;
    k = (int)(n * (CPU__main__createRandomArray__randInt__rand() / (RAND_MAX + 1.0)));
    return (k == n) ? k + min - 1 : k + min;
}

int randInt(int min, int max) {
    int k, n;
    n = (max - min) + 1;
    k = (int)(n * (rand() / (RAND_MAX + 1.0)));
    return (k == n) ? k + min - 1 : k + min;
}



//Generated clone.
void GPU__main__createRandomArray__shuffle(int* items, int len) {
    size_t j, k, i;
    int aux;

    for(i = len-1; i > 0; --i) {
        k = (int)((i + 1) * (CPU__main__createRandomArray__shuffle__rand() / (RAND_MAX + 1.0)));
        j = (k == (i + 1)) ? k - 1 : k;

        aux = items[i];
        items[i] = items[j];
        items[j] = aux;
    }
}



//Generated clone.
void CPU__main__createRandomArray__shuffle(int* items, int len) {
    size_t j, k, i;
    int aux;

    for(i = len-1; i > 0; --i) {
        k = (int)((i + 1) * (CPU__main__createRandomArray__shuffle__rand() / (RAND_MAX + 1.0)));
        j = (k == (i + 1)) ? k - 1 : k;

        aux = items[i];
        items[i] = items[j];
        items[j] = aux;
    }
}

void shuffle(int* items, int len) {
    size_t j, k, i;
    int aux;

    for(i = len-1; i > 0; --i) {
        k = (int)((i + 1) * (rand() / (RAND_MAX + 1.0)));
        j = (k == (i + 1)) ? k - 1 : k;

        aux = items[i];
        items[i] = items[j];
        items[j] = aux;
    }
}



//Generated clone.
int *GPU__main__createRandomArray(int size) {
    int i, len;
    int *result;

    len = size + 1;
    result = (int*)malloc(len * sizeof(int));
    long long int AI1[4];
    AI1[0] = 4 * size;
    AI1[1] = AI1[0] / 4;
    AI1[2] = (AI1[1] > 0);
    AI1[3] = (AI1[2] ? AI1[1] : 0);
    #pragma acc data pcopy(result[0:AI1[3]])
    #pragma acc kernels
    for (i = 0; i < len; i++)
        result[i] = i;
    result[0] = GPU__main__createRandomArray__randInt(1, size);
    GPU__main__createRandomArray__shuffle(result, len);
    return result;
}



//Generated clone.
int *CPU__main__createRandomArray(int size) {
    int i, len;
    int *result;

    len = size + 1;
    result = (int*)malloc(len * sizeof(int));
    for (i = 0; i < len; i++)
        result[i] = i;
    result[0] = CPU__main__createRandomArray__randInt(1, size);
    CPU__main__createRandomArray__shuffle(result, len);
    return result;
}

int *createRandomArray(int size) {
    int i, len;
    int *result;

    len = size + 1;
    result = (int*)malloc(len * sizeof(int));
    for (i = 0; i < len; i++)
        result[i] = i;
    result[0] = randInt(1, size);
    shuffle(result, len);
    return result;
}



//Generated clone.
int GPU__main__findDuplicate(int *data, int len) {
    int i;
    int result = 0;

    long long int AI1[5];
    AI1[0] = len + -1;
    AI1[1] = 4 * AI1[0];
    AI1[2] = AI1[1] / 4;
    AI1[3] = (AI1[2] > 0);
    AI1[4] = (AI1[3] ? AI1[2] : 0);
    #pragma acc data pcopy(data[0:AI1[4]])
    #pragma acc kernels
    for (i = 0; i < len; i++)
        result = result ^ (i + 1) ^ data[i];
    result ^= len;
    return result;
}



//Generated clone.
int CPU__main__findDuplicate(int *data, int len) {
    int i;
    int result = 0;

    for (i = 0; i < len; i++)
        result = result ^ (i + 1) ^ data[i];
    result ^= len;
    return result;
}

int findDuplicate(int *data, int len) {
    int i;
    int result = 0;

    for (i = 0; i < len; i++)
        result = result ^ (i + 1) ^ data[i];
    result ^= len;
    return result;
}



//Generated clone.
int GPU__main() {
    int i, j, duplicate;
    int *rndArr;

    GPU__main__srand(1);

	for (i = 0; i < NLOOPS1; i++) {
		rndArr = GPU__main__createRandomArray(ARRAY_SIZE);
		for (j = 0; j < NLOOPS2; j++)
			duplicate = GPU__main__findDuplicate(rndArr, ARRAY_SIZE+1);
		free(rndArr);
		printf("Found duplicate: %d\n", duplicate);
	}

    return 0;
}

int main() {
    int i, j, duplicate;
    int *rndArr;

    CPU__main__srand(1);

	for (i = 0; i < NLOOPS1; i++) {
		rndArr = GPU__main__createRandomArray(ARRAY_SIZE);
		for (j = 0; j < NLOOPS2; j++)
			duplicate = CPU__main__findDuplicate(rndArr, ARRAY_SIZE+1);
		free(rndArr);
		printf("Found duplicate: %d\n", duplicate);
	}

    return 0;
}

