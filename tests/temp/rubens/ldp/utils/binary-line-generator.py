#!/usr/bin/python


from sys import argv
from random import randint, sample


num_lines = int(argv[1])
line_len = int(argv[2])
pattern_len = int(argv[3])
prob = float(argv[4]) / 100

prob = min(prob, 100)
line_len = max(line_len, pattern_len)

pattern = bin(randint(1 << pattern_len, 2 << pattern_len))[-pattern_len:]

num_occur = int(prob * num_lines)
matches = sample(xrange(0, num_lines), num_occur)

# Printing arguments
print(num_lines)
print(pattern)

# Printing binary lines
matches.sort(reverse=True)
for i in xrange(num_lines):

	line = ""
	while len(line) < line_len:
		line += bin(randint(0, 2 << 100))[2:]

	if matches and matches[-1] == i:
		line += pattern
		matches.pop()

	print(line[-line_len:])
