#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../../include/time_common.h"
// #define DEBUG
#define CHECK_RESULTS

typedef struct Book Book;

struct Book {
  int numLines;
  unsigned long long int numCharsPerLine;
  char **lines;
};

Book *readBook(int lines, unsigned long long int chars);

void printBook(Book *b);

void freeBook(Book *b);

void filterLines(Book b, char *word, int wordSize, int *occurrences,
                 int *alphabet);

void filterLine(char *line, int lineSize, char *word, int wordSize,
                int *occurrences, int *alphabet);

int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("Not enough arguments to main. <LINES> <SIZE> <WORD>  \n");
    return 0;
  }

  Instance *I = newInstance(100);
  clock_t beg, end;
  int lines = atoi(argv[1]);
  unsigned long long chars = atoi(argv[2]);

  Book *book;
  if ((book = readBook(lines, chars)) == NULL) {
    printf("Could not open file.\n");
    return 0;
  }

  int *filtered = (int *)malloc(book->numLines * sizeof(int));
  int *alphabet = (int *)malloc(book->numLines * sizeof(int));
  memset(filtered, 0, sizeof(int) * book->numLines);
  memset(alphabet, 0, sizeof(int) * book->numLines);

  char *word = argv[3];
  int wordSize = strlen(word);

#ifdef DEBUG
  printBook(book);
#endif

  beg = clock();
  filterLines(*book, word, wordSize, filtered, alphabet);
  end = clock();

  addNewEntry(I, book->numCharsPerLine, getTimeInSecs(end - beg));

#ifdef CHECK_RESULTS
  // debugging
  // printLines(lines, numLines, numChars);
  for (int i = 0; i < book->numLines; i++) {
    printf("Found %d matches and %d alphabet sequences in line %d\n",
           filtered[i], alphabet[i], i);
  }
#endif

  writeResultsToOutput(stdout, I);
  freeInstance(I);

  freeBook(book);
  free(filtered);
  free(alphabet);

  return 0;
}

void filterLines(Book b, char *word, int wordSize, int *occurrences,
                 int *alphabet) {
  for (int i = 0; i < b.numLines; i++) {
    filterLine(b.lines[i], b.numCharsPerLine, word, wordSize, &occurrences[i],
               &alphabet[i]);
  }
}

void filterLine(char *line, int lineSize, char *word, int wordSize,
                int *occurrences, int *alphabet) {
  for (int i = 0; i < lineSize; i++) {
    if (*(line + i) == word[0]) // found first letter
    {
      for (int k = 1; k < wordSize; k++) {
        if (i + k >= lineSize)
          break;

        if (*(line + i + k) != word[k])
          break;

        if (k == wordSize - 1)
          (*occurrences)++;
      }
    }
  }

  int a, b;

  for (int i = 0; i < lineSize - 1; i++) {
    a = (int)(*(line + i));
    b = (int)(*(line + i + 1));

    // printf("%d %d\n", a, b);

    if (a == b + 1)
      (*alphabet)++;
  }

  for (int i = 0 + *occurrences; i < lineSize; i++)
    ;
}

Book *readBook(int lines, unsigned long long int chars) {
  Book *book = (Book *)malloc(sizeof(Book));
  book->numLines = lines;
  book->numCharsPerLine = chars;
  book->lines = (char **)malloc(sizeof(char *) * book->numLines);

  srand(time(NULL));
  int i;
  for (i = 0; i < (book->numLines); i++) {
    book->lines[i] = (char *)malloc(sizeof(char) * book->numCharsPerLine);
    for (int j = 0; j < book->numCharsPerLine; j++) {
      book->lines[i][j] = 'a' + (rand() % 26);
    }
  }
  return book;
}

void printBook(Book *b) {
  int i;
  for (i = 0; i < b->numLines; i++) {
    printf("%s\n", b->lines[i]);
  }
}

void freeBook(Book *b) {
  int i;
  for (i = 0; i < b->numLines; i++) {
    free(b->lines[i]);
  }
  free(b->lines);
  free(b);
}
