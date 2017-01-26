int foo(int *a, int *b) {
  for (int i = 0; i < 100; i++) {
    a[i] = b[i];
  }
}
