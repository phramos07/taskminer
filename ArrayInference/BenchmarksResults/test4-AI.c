#include<stdio.h>
#include<stdlib.h>

void func(int **v) {
  int i,j;
  int* v2;
// Using the same base pointer
// This occurs in bug.
// I'm using the type defined in LLVM, but when I've 
// a bitcast two dimensional array, the pass will go
// return a bidimensional result, but I need of the
// last bitcast used? (I really don't know.)
// Remember that haven't ambiguation in the pointer or something.
  v2 = (int*) v;
  for (i = 0; i < 10000; i++)
      printf("%d => %d\n",i,v2[i]);
}

int main() {
  int i,j;
  int v[100][100];
  acc_create((void*) v, 40400);
  acc_copyin((void*) v, 40400);
  for (i = 0; i < 100; i++)
    for (j = 0; j < 100; j++)
      v[i][j] = 1;
  func(v);
  acc_copyout_and_keep((void*) v, 40400);
  return 0;
}

