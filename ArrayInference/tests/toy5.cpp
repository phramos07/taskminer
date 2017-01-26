#include <stdio.h>

int main() {
  int N = 10;
  int vet[N];

  /// Initialize array. This is a DOALL loop
  for (int i = 0; i < N; i++)
    vet[i] = i;

  /// Just print the array
  for (int i = 0; i < N; i++)
    printf("vet[%d] = %d\n", i, vet[i]);

  return 0;
}
