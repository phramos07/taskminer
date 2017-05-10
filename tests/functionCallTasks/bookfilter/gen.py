import sys
import random

NUM_LINES_DEFAULT=10
NUM_CHARS_DEFAULT=80000

def genBook(fileName, numLines=NUM_LINES_DEFAULT, numChars=NUM_CHARS_DEFAULT):
	with open(fileName, "w") as f:
		f.write(str(numLines) + "\n");
		f.write(str(numChars) + "\n")
		for i in range(0, numLines):
			for j in range(0, numChars):
				f.write(chr((random.randint(97, 122))))
			f.write("\n");
		f.write("\n");

if __name__ == '__main__':
	genBook(sys.argv[1]);