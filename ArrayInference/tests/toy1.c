/*
Toy Benchmark #1: INS / OUTS test

Test whether the taskminer is getting right the INs and OUTs sets.

*/

#include <stdio.h>

int main() {
  int N = 10;
  int vet[N];
  int array[N];

  /// Initialize array. This is a DOALL loop - WRITE
  for (int i = 0; i < N; i++) {
    vet[i] = i;
    array[i] = i;
  }

  /// Just print the array - READ
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      printf("vet[%d] = %d\n", i, vet[j]);

  /// READ_WRITE vet, READ array
  for (int k = 0; k < N; k++) {
    printf("vet[%d] = %d\n", k, vet[k]);
    vet[k] = vet[k] + k * k;

    printf("array[%d] = %d\n", k, array[k]);
  }

  return 0;
}
