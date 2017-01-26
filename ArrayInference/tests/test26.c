void foo(int *v, int *s, int n, int m) {

  for (int i = 0; i < n; i++)
    v[i] = s[i];

  for (int j = 0; j < m; j++)
    s[j] = v[j];

  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
      v[i * m + j] = s[j] + s[i];
}

void foo2(int n, int *vect) {

  for (int i = 0; i < n; i++)
    vect[i] = vect[i] + 1;
}
