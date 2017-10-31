int f(int *v) {
  for (int i = 0; i < 500; i++) {
    v[i] *= 2;
  }
  return 0;
}

int main() {
  int v[500];
  return f(v);
}
