int foo(int argc, int *y) {
  for (int i = 0; i < argc; ++i) {
    int j = *y;
    *y = j + 1;
  }
  return 0;
}
