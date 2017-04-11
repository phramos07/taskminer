#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#define DEBUG

struct Line
{
	char* line;
	int size;

}typedef Line;

Line* getLines(const char* name, int* numLines);

void printLines(Line* line);

void filterLine(const Line line, const char* word, int wordSize, int* occurrences);

int main(int argc, char const *argv[])
{
	omp_set_dynamic(1);
	if (argc < 3)
	{
		printf("Not enough arguments to main. <BOOK> <WORD> \n");
		return 0;
	}
	int numLines;
	Line* lines = getLines(argv[1], &numLines);

	int* filtered = (int*)malloc(numLines*sizeof(int));

	const char* word = argv[2];
	const int wordSize = strlen(word);

	#pragma omp parallel
	#pragma omp single
	for (int i = 0; i < numLines; i++)
	{
		#pragma omp task depend(in:lines[i])
		filterLine(lines[i], word, wordSize, &filtered[i]);
	}

	#ifdef DEBUG
		//debugging
		// printLines(lines);
		for (int i = 0; i < numLines; i++)
		{
			printf("Found %d matches in line %d\n", filtered[i], i);
		}
	#endif

	return 0;
}

void printLines(Line* line)
{
	for (; line->line != NULL; line++)
	{
		printf("%s\n", line->line);
	}
}

Line* getLines(const char* name, int* numLines)
{
	Line* lines;
	int numChars;
	FILE* in;
	in = fopen(name, "r");
	fscanf(in, "%d\n", &(*numLines));
	lines = (Line*)malloc((*numLines)*sizeof(Line));
	for (int i = 0; i < (*numLines); i++)
	{
		fscanf(in, "%d ", &numChars);
		lines[i].size = numChars;
		lines[i].line = (char*)malloc((numChars+1)*sizeof(char));
		for (int j = 0; j < numChars; j++)
		{
			fscanf(in, "%c", &lines[i].line[j]);
		}
		fscanf(in, "\n", NULL);
	}
	// fscanf(in, "\n", NULL);

	return lines;
}


void filterLine(const Line l, const char* word, int wordSize, int* occurrences)
{
	for (int i = 0; i < l.size; i++)
	{
		if(l.line[i] == word[0]) //found first letter
		{
			for (int k = 1; k < wordSize; k++)
			{
				if (i+k >= l.size)
					break;

				if (l.line[i+k] != l.line[k])
					break;

				if (k == wordSize-1)
					(*occurrences)++;
			}
		}
	}
}








