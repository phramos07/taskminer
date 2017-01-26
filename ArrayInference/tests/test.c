void fdtd_2d(float *Y, float *X, float *H, int m, int n) {
  int i = 0;
  if (shouldComputY(i)) {
    for (int j = 0; j < n; j++)
      Y[i * n + j] = Y[i * n + j] - 0.5 * (H[i * n + j] - H[(i - 1) * n + j]);
  }

  for (int j = 1; j < n; j++)
    X[i * n + j] = X[i * n + j] - 0.5 * (H[i * n + j] - H[i * n + (j - 1)]);
}
