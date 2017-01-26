int *v;

void foo(int *p, int n) {
  int i;

  for (i = 0; i < n; i++) {
    if (i == 0)
      p[i] = v[i];

    v[i] = i;
  }

  printf("%d", v[0]);
}

void main() {
  v = malloc(sizeof(int) * 10);
  int *valid = malloc(sizeof(int) * 10);
  valid[0] = 42;
  v[0] = valid;

  int *p;
  p = &v;

  foo(p, 2);
}
