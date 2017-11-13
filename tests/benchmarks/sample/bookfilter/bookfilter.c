#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#define DEBUG

typedef struct Book Book;

struct Book
{
	int numLines;
	int numCharsPerLine;
	char** lines;
};

void filterLines(Book b, char* word,  int wordSize, int* occurrences, int* alphabet);

void filterLine( char* line,  int lineSize,  char* word,  int wordSize, int* occurrences, int* alphabet) { return; }

void filterLines(Book b, char* word,  int wordSize, int* occurrences, int* alphabet)
{
	#pragma omp parallel
	#pragma omp single
	for (int i = 0; i < b.numLines; i++)
	{
		#pragma omp task depend(in:b, occurrences[i], alphabet[i])
		filterLine(b.lines[i], b.numCharsPerLine, word, wordSize, &occurrences[i], &alphabet[i]);
	}
}

int main(int argc, char const *argv[])
{
	return 0;
}

