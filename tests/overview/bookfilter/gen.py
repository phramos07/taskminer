import sys
import random

NUM_LINES_DEFAULT=10
NUM_CHARS_DEFAULT=100
VARIATION=100

def genBook(fileName, numLines=NUM_LINES_DEFAULT, numChars=NUM_CHARS_DEFAULT):
	with open(fileName, "w") as f:
		f.write(str(numLines) + "\n");
		for i in range(0, numLines):
			numChars_ = random.randint(numChars, numChars+VARIATION)
			f.write(str(numChars_) + " ")
			for j in range(0, numChars_):
				f.write(chr((random.randint(97, 122))))
			f.write("\n");
		f.write("\n");

if __name__ == '__main__':
	genBook(sys.argv[1]);