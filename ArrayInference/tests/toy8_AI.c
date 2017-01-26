#include <stdio.h>
#include <stdlib.h>

void one_read(int *a) {
  *a = 100;
  printf("[one_read] a = %d\n", *a);
}

void task(int *v, int i) {
  v[i] = v[i] & 0x0000FFFF;
  printf("%d\n", v[i]);
}

int main() {
  const int N = 100000;
  int u[N];
  int *v;
  int *test;
  v = (int *)malloc(20 * sizeof(int));
  test = (int *)malloc(20 * sizeof(int));
  int sum = 0;

  #pragma omp parallel
  #pragma omp single
  for (int i = 0; i < N; i++) {

    long long int TM1[2];
    TM1[0] = i * 4;
    TM1[1] = (TM1[0] / 4);
    #pragma omp task depend(inout:u[TM1[1]])
    one_read(&u[i]);
    sum += v[i];
    int tmp = test[i] * test[i];
    tmp ^= i;
    sum += tmp;
    #pragma omp task depend(inout:v,i)
    task(v, i);
  }

  // printf("Finished. %d\n", u[rand() % N]);
  return sum;
}

