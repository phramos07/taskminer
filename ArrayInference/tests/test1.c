#include <stdlib.h>
typedef struct var {
  long int a;
  long int b;
  long int c[100];
} var;

int main(int argc, char *argv[]) {
  int b = atoi(argv[1]);
  // int b = 10000;
  var *n = (var *)malloc(sizeof(var) * b);
  // int *n = (int*) malloc(sizeof(int) * b);
  for (int i = 0; i < b; i++) {
    for (int j = 0; j < b; j++)
      n[i].a = 0;
    n[i].b = i;
  }

  for (int i = 0; i <= b; i++)
    n[i + 2].a = n[i + 2].b + 1;

  return 0;
}
