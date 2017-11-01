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

Book* readBook(char* filename);

void printBook(Book* b);

void freeBook(Book* b);

void filterLines(Book b, char* word,  int wordSize, int* occurrences, int* alphabet);

void filterLine( char* line,  int lineSize,  char* word,  int wordSize, int* occurrences, int* alphabet);

int main(int argc, char  *argv[])
{
	if (argc < 3)
	{
		printf("Not enough arguments to main. <BOOK> <WORD> \n");
		return 0;
	}
	Book* book;
	if ((book = readBook(argv[1])) == NULL)
	{
		printf("Could not open file.\n");
		return 0;
	}

	int* filtered = (int*)malloc(book->numLines*sizeof(int));
	int* alphabet = (int*)malloc(book->numLines*sizeof(int));
	memset(filtered, 0, sizeof(int)*book->numLines);
	memset(alphabet, 0, sizeof(int)*book->numLines);

	char* word = argv[2];
	int wordSize = strlen(word);

	#ifdef DEBUG
		printBook(book);
	#endif


	#ifdef DEBUG
		//debugging
		// printLines(lines, numLines, numChars);
		for (int i = 0; i < book->numLines; i++)
		{
			printf("Found %d matches and %d alphabet sequences in line %d\n", filtered[i], alphabet[i],i);
		}
	#endif

	freeBook(book);
	free(filtered);
	free(alphabet);

	return 0;
}

void filterLines(Book b, char* word,  int wordSize, int* occurrences, int* alphabet)
{
	for (int i = 0; i < b.numLines; i++)
	{
		filterLine(b.lines[i], b.numCharsPerLine, word, wordSize, &occurrences[i], &alphabet[i]);
	}
}

void filterLine(char* line, int lineSize, char* word,  int wordSize, int* occurrences, int* alphabet)
{
	for (int i = 0; i < lineSize; i++)
	{
		if(*(line + i) == word[0]) //found first letter
		{
			for (int k = 1; k < wordSize; k++)
			{
				if (i+k >= lineSize)
					break;

				if (*(line + i + k) != word[k])
					break;

				if (k == wordSize-1)
					(*occurrences)++;
			}
		}
	}

	int a, b;

	for (int i= 0; i < lineSize-1; i++)
	{
		a = (int)(*(line + i));
		b = (int)(*(line + i + 1));

		// printf("%d %d\n", a, b);

		if (a == b+1)
			(*alphabet)++;		
	}

	// for (int i = 0 + *occurrences; i < lineSize; i++)
	// 	for (int j = *occurrences; j < lineSize; j++);
}


Book* readBook(char* filename)
{
	FILE* in = fopen(filename, "r");
	
	if (!in)
		return NULL;

	Book* book = (Book*) malloc(sizeof(Book));
	fscanf(in, "%d\n", &book->numLines);
	fscanf(in, "%d\n", &book->numCharsPerLine);
	book->lines = (char**) malloc(sizeof(char*) * book->numLines);
	
	int i;
	for (i = 0; i < (book->numLines); i++)
	{
		book->lines[i] = (char*) malloc(sizeof(char) * book->numCharsPerLine);
		for (int j = 0; j < book->numCharsPerLine; j++)
		{
			fscanf(in, "%c", &book->lines[i][j]);
		}
		fscanf(in, "\n", NULL);
	}
	

	return book;
}

void printBook(Book* b)
{
	int i;
	for (i = 0; i < b->numLines; i++)
	{
		printf("%s\n", b->lines[i]);
	}
}

void freeBook(Book* b)
{
	int i;
	for (i = 0; i < b->numLines; i++)
	{
		free(b->lines[i]);
	}	
	free(b->lines);
	free(b);
}



