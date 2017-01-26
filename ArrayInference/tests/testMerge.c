#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#include <stdio.h>

#define TAM 1000
#define NUM_THREADS 4
#define ITERS 10

int* genRandomVector(int size);

void printVector(int *v, int size);

void merge_sort_omp(int* vec, int size, int *tmp);

void merge_sort_serial(int* vec, int size, int *tmp);

void merge(int *vec, int size, int* tmp);

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        printf("Must input the size of the array and number of iterations\n");
        return 0;
    }
    // omp_set_num_threads(NUM_THREADS);
    // omp_set_nested(1);
    int size = atoi(argv[1]);
    int iters = atoi(argv[2]);
    int *v = genRandomVector(size);
    int tmp[size];
    for (int i=0; i < ITERS; i++)
    {
        #pragma omp parallel
        {
            #pragma omp single
            {
                printVector(v, size);
                merge_sort_omp(v, size, tmp);
                printf("\n\n");
                printVector(v, size);       
            }
        }
        free(v);
        v = genRandomVector(size);
    }

    return 0;
}

void printVector(int *v, int size)
{
    int* p = v;
    for (int i=0; i<size; i++)
    {
        printf("%d ", *p);
        p++;
    }
    printf("\n");
}

int* genRandomVector(int size)
{
    int*vec = (int*)malloc(sizeof(int)*size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = (rand()%size);
    }
    return vec;
}

void merge_sort_serial(int* vec, int size, int *tmp)
{
    if(size == 1)
    {
        return;
    }
    merge_sort_serial(vec, size/2, tmp);
        merge_sort_serial(vec + (size/2), size-(size/2), tmp);
    merge(vec, size, tmp);
}

void merge_sort_omp(int* vec, int size, int *tmp)
{
    // int ID = omp_get_thread_num();
 //    printf("Thread %d\n", ID);
    if(size == 1)
    {
        return;
    }
    #pragma omp task firstprivate(vec, size, tmp)
    merge_sort_omp(vec, size/2, tmp);
    #pragma omp task firstprivate(vec, size, tmp)
    merge_sort_omp(vec + (size/2), size-(size/2), tmp);

    #pragma omp taskwait
    merge(vec, size, tmp);
}

void merge(int *vec, int size, int* tmp)
{
    int left_it = 0;
    int right_it = size/2;
    int ti=0;

    while(left_it < size/2 && right_it < size)
    {
        if(vec[left_it] < vec[right_it])
        {
            tmp[ti] = vec[left_it];
            left_it++;
            ti++;
        }
        else
        {
            tmp[ti] = vec[right_it];
            right_it++;
            ti++;
        }
    }
    while(left_it < size/2)
    {
        tmp[ti] = vec[left_it];
        left_it++;
        ti++;
    }
    while(right_it < size)
    {
        tmp[ti] = vec[right_it];
        right_it++;
        ti++;
    }
    memcpy(vec, tmp, size*sizeof(int));
}
