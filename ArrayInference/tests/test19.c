const int N = 500;

void f(int v[N]) {
  for (int i = 0; i < N; i++)
    v[i] *= 2;
}

int main() {
  int v[N];
  f(v);
  return 0;
}

