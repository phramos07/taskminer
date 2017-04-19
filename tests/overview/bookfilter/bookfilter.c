#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#define DEBUG

// Line* getLines(const char* name, int* numLines);

char** getLines(const char* name, int* numLines, int* size);

void printLines(char** lines, const int numLines);

// void filterLine(const Line l, const char* word, int wordSize, int* occurrences, int* alphabet);

void filterLine(const char* line, const int lineSize, const char* word, const int wordSize, int* occurrences, int* alphabet);

int main(int argc, char const *argv[])
{
	if (argc < 3)
	{
		printf("Not enough arguments to main. <BOOK> <WORD> \n");
		return 0;
	}

	int numLines, numChars;
	int* size;
	// char** lines = getLines(argv[1], &numLines, &(*size));

	char** lines;
	FILE* in;
	in = fopen(argv[1], "r");
	fscanf(in, "%d\n", &(numLines));
	size = (int*)malloc((numLines)*sizeof(int));
	lines = (char**)malloc((numLines)*sizeof(char*));
	for (int i = 0; i < (numLines); i++)
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

	int* filtered = (int*)malloc(numLines*sizeof(int));
	int* alphabet = (int*)malloc(numLines*sizeof(int));

	const char* word = argv[2];
	const int wordSize = strlen(word);

	// #pragma omp parallel
	// #pragma omp single
	for (int i = 0; i < numLines; i++)
	{
		// #pragma omp task depend(in:lines[i], size[i]) depend(out: filtered[i], alphabet[i])
		filterLine(lines[i], size[i], word, wordSize, &filtered[i], &alphabet[i]);
	}

	#ifdef DEBUG
		//debugging
		// printLines(lines, numLines);
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

void printLines(char** lines, const int numLines)
{
	int i;
	for (i=0; i < numLines; i++)
	{
		printf("%s\n", lines[i]);
	}
}

// Line* getLines(const char* name, int* numLines)
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

char** getLines(const char* name, int* numLines, int* size)
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


// void filterLine(const Line l, const char* word, int wordSize, int* occurrences, int* alphabet)
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


void filterLine(const char* line, const int lineSize, const char* word, const int wordSize, int* occurrences, int* alphabet)
{
	for (int i = 0; i < lineSize; i++)
	{
		if(line[i] == word[0]) //found first letter
		{
			for (int k = 1; k < wordSize; k++)
			{
				if (i+k >= lineSize)
					break;

				if (line[i+k] != line[k])
					break;

				if (k == wordSize-1)
					(*occurrences)++;
			}
		}
	}

	int a, b;

	for (int i= 0; i < lineSize; i++)
	{
		a = (int)(line[i]);
		b = (int)(line[i+1]);

		// printf("%d %d\n", a, b);

		if (a == b+1)
			(*alphabet)++;		
	}

	for (int i = 0 + *occurrences; i < lineSize; i++)
		for (int j = *occurrences; j < lineSize; j++);
}





