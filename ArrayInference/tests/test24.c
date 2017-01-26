int *v;

void foo(int n) {
  int i = 0;
  for (i = 0; i < n; i++)
    v[i] = v[i] + i;
}
