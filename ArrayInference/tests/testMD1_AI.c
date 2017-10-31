
void foo(int **V) {
  int i, j;
  long long int TM1[2];
  TM1[0] = i + 1;
  TM1[1] = j + 1;
  #pragma omp parallel
  #pragma omp single
  #pragma omp task depend(inout:V[TM1[0]],V[TM1[0]][TM1[1]])
  {
  for (i = 0; i < 100; i++)
    for (j = 0; j < 100; j++)
      V[i + 1][j + 1] = i + j;
}
}

