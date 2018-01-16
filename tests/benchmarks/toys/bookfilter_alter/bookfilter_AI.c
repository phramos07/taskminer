#include <omp.h>
#ifndef taskminerutils
#define taskminerutils
static int taskminer_depth_cutoff = 0;
#define DEPTH_CUTOFF omp_get_num_threads()
char cutoff_test = 0;
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#define DEBUG

// Line* getLines(char* name, int* numLines);

char **getLines(char *name, int *numLines, int *size);

void printLines(char *lines, int numLines, int numChars);

void filterLines(char *lines, int *size, int numLines, int numChars, char *word,
                 int wordSize, int *occurrences, int *alphabet);

void filterLine(char *line, int lineSize, char *word, int wordSize,
                int *occurrences, int *alphabet);

int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("Not enough arguments to main. <BOOK> <WORD> \n");
    return 0;
  }

  int numLines, numChars;
  int *size;
  // char** lines = getLines(argv[1], &numLines, &(*size));

  char *lines;
  numLines = atoi(argv[1]);
  numChars = atoi(argv[2]);
  size = (int *)malloc((numLines) * sizeof(int));
  lines = (char *)malloc((numLines * numChars + 1) * sizeof(char *));
  for (int i = 0; i < (numLines); i++) {
    size[i] = numChars;
    // lines[i] = (char*)malloc((numChars+1)*sizeof(char));
    for (int j = 0; j < numChars; j++) {
      lines[i * numChars + j] = 'a' + (rand() % 26);
    }
  }

  int *filtered = (int *)malloc(numLines * sizeof(int));
  int *alphabet = (int *)malloc(numLines * sizeof(int));
  #pragma omp parallel
  #pragma omp single
  for (int i = 0; i < numLines; i++) {
    long long int TM31[7];
    TM31[0] = numLines > 0;
    TM31[1] = (TM31[0] ? numLines : 0);
    TM31[2] = 4 * TM31[1];
    TM31[3] = TM31[2] + 1;
    TM31[4] = TM31[3] / 4;
    TM31[5] = (TM31[4] > 0);
    TM31[6] = (TM31[5] ? TM31[4] : 0);
    #pragma omp task depend(inout: alphabet[0:TM31[6]],filtered[0:TM31[6]])
    {
    filtered[i] = 0;
    alphabet[i] = 0;
  }
  }

  char *word = argv[2];
  int wordSize = strlen(word);

  // printLines(lines, numLines, numChars);

  filterLines(lines, size, numLines, numChars, word, wordSize, filtered,
              alphabet);

// for (int i = 0; i < numLines; i++)
// {
// 	filterLine(&lines[i*numChars], size[i], word, wordSize, &filtered[i],
// &alphabet[i]);
// }

#ifdef DEBUG
  // debugging
  // printLines(lines, numLines, numChars);
  for (int i = 0; i < numLines; i++) {
    printf("Found %d matches and %d alphabet sequences in line %d\n",
           filtered[i], alphabet[i], i);
  }
#endif

  // free(filtered);
  // free(alphabet);
  // free(lines);
  // free(size);

  return 0;
}

void printLines(char *lines, int numLines, int numChars) {
  int i, j;
  for (i = 0; i < numLines; i++) {
    for (j = 0; j < numChars; j++)
      printf("%c", lines[i * numChars + j]);
    printf("\n");
  }
}

char **getLines(char *name, int *numLines, int *size) {
  char **lines;
  int numChars;
  FILE *in;
  in = fopen(name, "r");
  fscanf(in, "%d\n", &(*numLines));
  size = (int *)malloc((*numLines) * sizeof(int));
  lines = (char **)malloc((*numLines) * sizeof(char *));
  for (int i = 0; i < (*numLines); i++) {
    fscanf(in, "%d ", &numChars);
    size[i] = numChars;
    lines[i] = (char *)malloc((numChars + 1) * sizeof(char));
    for (int j = 0; j < numChars; j++) {
      fscanf(in, "%c", &lines[i][j]);
    }
    fscanf(in, "\n", NULL);
  }

  return lines;
}

void filterLines(char *lines, int *size, int numLines, int numChars, char *word,
                 int wordSize, int *occurrences, int *alphabet) {
  #pragma omp parallel
  #pragma omp single
  for (int i = 0; i < numLines; i++) {
    int size_ = size[i];
    cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
    long long int TM11[1];
    TM11[0] = i * numChars;
    #pragma omp task untied default(shared) depend(in:word,lines[TM11[0]],size[i]) depend(inout:alphabet[i],occurrences[i]) private(lines[TM14[0]],alphabet[i],size[i],occurrences[i])
    filterLine(&lines[i * numChars], size[i], word, wordSize, &occurrences[i],
               &alphabet[i]);
  }
#pragma omp taskwait
}

void filterLine(char *line, int lineSize, char *word, int wordSize,
                int *occurrences, int *alphabet) {
  for (int i = 0; i < lineSize; i++) {
    // found first letter
    if (*(line + i) == word[0]) {
      for (int k = 1; k < wordSize; k++) {
        if (i + k >= lineSize)
          break;

        if (*(line + i + k) != word[k])
          break;

        if (k == wordSize - 1) {
          (*occurrences)++;
        }
      }
    }
  }

  int a, b;

  for (int i = 0; i < lineSize - 1; i++) {
    a = (int)(*(line + i));
    b = (int)(*(line + i + 1));

    // printf("%d %d\n", a, b);

    if (a == b + 1) {
      (*alphabet)++;
    }
  }

  // for (int i = 0 + *occurrences; i < lineSize; i++) {
  //   for (int j = *occurrences; j < lineSize; j++) {
  //   }
  // }
}

