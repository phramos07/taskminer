import re

filename = raw_input("Input filename:")

f = open(filename, 'r')

string = f.read()

benchs = re.compile(r'Running (.*) -*')
times = re.compile(r'real\t0m(.*)s')

benchlist = benchs.findall(string)
timelist = times.findall(string)

index = 0
while index < len(benchlist):
	bench = benchlist[index]

	time = 0;
	for j in range(5):
		time = time + float(timelist[(index*5) + j])


	average = time/5

	print "bench: " + str(bench) + "\ttime: " + str(average)

	index = index + 1
