int foo(int *a, int *b) {
  char RST_AI1 = 0;
  RST_AI1 |= !((a + 0 > b + 100)
  || (b + 0 > a + 100));
  #pragma acc data pcopyin(b[0:100]) pcopyout(a[0:100]) if(!RST_AI1)
  #pragma acc kernels if(!RST_AI1)
  #pragma acc loop independent 
  for (int i = 0; i < 100; i++) {
    a[i] = b[i];
  }
}

