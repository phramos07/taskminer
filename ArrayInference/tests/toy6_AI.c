#include <stdio.h>
#include <stdlib.h>

#define PLANE_SIZE 50
static char plane[PLANE_SIZE];

void fill() {
  /* Fill the plane with the bit mask for remainders */
  unsigned i;
  for (i = 0; i < PLANE_SIZE; i += 2) {
    plane[i] = 0;
    plane[i + 1] = 1;
  }
}

int main() {
  unsigned i;

  fill();

  for (i = 0; i < PLANE_SIZE; i++) {
    if (plane[i] & 0x10)
      printf("%d ", i);
  }
  printf("\n");

  return 0;
}

