void foo(int *a, int *b, int H, int n, int m) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
      a[i * H + j] = a[(i - 1) * H + j] + b[i * H + (j - 1)];
}

