
void foo(int **V) {
  int i, j;
  for (i = 0; i < 100; i++)
    for (j = 0; j < 100; j++)
      V[i + 1][j + 1] = i + j;
}
