#include <stdio.h>

#define test test2

void test(int x) {
  printf("%d\n", x);
}

int main(int argc, char* argv[]) {
  test(10);
}

