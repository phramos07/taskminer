
// Original:
/*int foo(int *a, int argc, char **argv) {
  // int a[100];
  int i, sum = 0;

  for (i = 0; i < argc; i++)
    a[i] = i;

  for (i = 0; i < argc; i++)
    sum += a[i];
}*/

void hotspot(float a, float *src, int x) {
  x = 3;
  for (int i = 0; i < 4; i++) {
    if (x == 0)
      src[i] = a * src[i];
    if (x == 1)
      src[i + 4] = a * src[i + 4];
    if (x == 2)
      src[i + 8] = a * src[i + 8];
    if (x == 3)
      src[i + 12] = a * src[i + 12];
  }
}

/*void saxpy(float a, float *x, float *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = a * x[i] + y[i];
}

float saxpy1(float a, float *x, float *y, int n) {
  int j = 0;

  for (int i = 0; i < n; ++i) {
    y[j] = a * x[i] + y[i];
    ++j;
  }
}*/
