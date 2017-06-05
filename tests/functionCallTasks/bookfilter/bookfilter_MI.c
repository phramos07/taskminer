#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#define DEBUG

// Line* getLines( char* name, int* numLines);

char** getLines( char* name, int* numLines, int* size);

void printLines(char* lines,  int numLines,  int numChars);

// void filterLine( Line l,  char* word, int wordSize, int* occurrences, int* alphabet);

void filterLines(char* lines, int* size, int numLines, int numChars, char* word,  int wordSize, int* occurrences, int* alphabet);

void filterLine( char* line,  int lineSize,  char* word,  int wordSize, int* occurrences, int* alphabet);

int main(int argc, char  *argv[])
{
	omp_set_dynamic(1);
	if (argc < 3)
	{
		printf("Not enough arguments to main. <BOOK> <WORD> \n");
		return 0;
	}

	int numLines, numChars;
	int* size;
	// char** lines = getLines(argv[1], &numLines, &(*size));

	char* lines;
	FILE* in;
	in = fopen(argv[1], "r");
	fscanf(in, "%d\n", &(numLines));
	fscanf(in, "%d\n", &numChars);
	size = (int*)malloc((numLines)*sizeof(int));
	lines = (char*)malloc((numLines*numChars + 1)*sizeof(char*));
	for (int i = 0; i < (numLines); i++)
	{
		size[i] = numChars;
		// lines[i] = (char*)malloc((numChars+1)*sizeof(char));
		for (int j = 0; j < numChars; j++)
		{
			fscanf(in, "%c", &lines[i*numChars + j]);
		}
		fscanf(in, "\n", NULL);
	}

	int* filtered = (int*)malloc(numLines*sizeof(int));
	int* alphabet = (int*)malloc(numLines*sizeof(int));
	for (int i = 0;  i< numLines; i++)
	{
		filtered[i] = 0;
		alphabet[i] = 0;
	}

	 char* word = argv[2];
	 int wordSize = strlen(word);

	// printLines(lines, numLines, numChars);

 filterLines(lines, size, numLines, numChars, word, wordSize, filtered, alphabet);

	// for (int i = 0; i < numLines; i++)
	// {
	// 	filterLine(&lines[i*numChars], size[i], word, wordSize, &filtered[i], &alphabet[i]);
	// }

	#ifdef DEBUG
		//debugging
		// printLines(lines, numLines, numChars);
		for (int i = 0; i < numLines; i++)
		{
			printf("Found %d matches and %d alphabet sequences in line %d\n", filtered[i], alphabet[i],i);
		}
	#endif

	// free(filtered);
	// free(alphabet);
	// free(lines);
	// free(size);

	return 0;
}

void printLines(char* lines,  int numLines,  int numChars)
{
	int i, j;
	for (i=0; i < numLines; i++)
	{
		for (j = 0; j < numChars; j++)
			printf("%c", lines[i*numChars + j]);
		printf("\n");
	}
}

// Line* getLines( char* name, int* numLines)
// {
// 	Line* lines;
// 	int numChars;
// 	FILE* in;
// 	in = fopen(name, "r");
// 	fscanf(in, "%d\n", &(*numLines));
// 	lines = (Line*)malloc((*numLines)*sizeof(Line));
// 	for (int i = 0; i < (*numLines); i++)
// 	{
// 		fscanf(in, "%d ", &numChars);
// 		lines[i].size = numChars;
// 		lines[i].line = (char*)malloc((numChars+1)*sizeof(char));
// 		for (int j = 0; j < numChars; j++)
// 		{
// 			fscanf(in, "%c", &lines[i].line[j]);
// 		}
// 		fscanf(in, "\n", NULL);
// 	}
// 	// fscanf(in, "\n", NULL);

// 	return lines;
// }

char** getLines( char* name, int* numLines, int* size)
{
	char** lines;
	int numChars;
	FILE* in;
	in = fopen(name, "r");
	fscanf(in, "%d\n", &(*numLines));
	size = (int*)malloc((*numLines)*sizeof(int));
	lines = (char**)malloc((*numLines)*sizeof(char*));
	for (int i = 0; i < (*numLines); i++)
	{
		fscanf(in, "%d ", &numChars);
		size[i] = numChars;
		lines[i] = (char*)malloc((numChars+1)*sizeof(char));
		for (int j = 0; j < numChars; j++)
		{
			fscanf(in, "%c", &lines[i][j]);
		}
		fscanf(in, "\n", NULL);
	}

	return lines;
}


// void filterLine( Line l,  char* word, int wordSize, int* occurrences, int* alphabet)
// {
// 	for (int i = 0; i < l.size; i++)
// 	{
// 		if(l.line[i] == word[0]) //found first letter
// 		{
// 			for (int k = 1; k < wordSize; k++)
// 			{
// 				if (i+k >= l.size)
// 					break;

// 				if (l.line[i+k] != l.line[k])
// 					break;

// 				if (k == wordSize-1)
// 					(*occurrences)++;
// 			}
// 		}
// 	}

// 	int a, b;

// 	for (int i= 0; i < l.size; i++)
// 		for (int j = i+1; j < l.size-1; j++)
// 		{
// 			a = atoi(&l.line[i]);
// 			b = atoi(&l.line[j]);

// 			if (a == (b-1))
// 				(*alphabet)++;
// 		}

// 	// for (int i = 0 + *occurrences; i < l.size; i++)
// 	// 	for (int j = *occurrences; j < 5000; j++);

// }

void filterLines(char* lines, int* size, int numLines, int numChars, char* word,  int wordSize, int* occurrences, int* alphabet)
{
	#pragma omp parallel
	#pragma omp single
	for (int i = 0; i < numLines; i++)
	{
		#pragma omp task depend(in:lines[i*numChars], size[i]) depend(inout: occurrences[i], alphabet[i])
		filterLine(&lines[i*numChars], size[i], word, wordSize, &occurrences[i], &alphabet[i]);
	}
}

void filterLine( char* line,  int lineSize,  char* word,  int wordSize, int* occurrences, int* alphabet)
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

	for (int i = 0 + *occurrences; i < lineSize; i++)
		for (int j = *occurrences; j < lineSize; j++);
}





