import re

def multiplier (unit):
	if (unit == 'ms'):
		return 1

	if (unit == 'us'):
		return 0.001

filename = raw_input("Input file location:")

f = open(filename, 'r')

string = f.read()

paths = re.compile(r'command: ././[a-z]+/[a-z]+/[a-zA-Z0-9|\-]+/(.*)_(.*)_gpu')
mem = re.compile(r'.*%\s\s(.*)[us|ms]?.*\[CUDA memcpy')

benchlist = paths.findall(string)
timelist = mem.findall(string)

for i in range(len(timelist)):
	timelist[i] = timelist[i].split()[0]

print benchlist
print timelist

print "num of benchs: " + str(len(benchlist))
print "num of times: " + str(len(timelist))

'''index = 0
while index < len(benchlist):
	bench = benchlist[index][0]
	size = benchlist[index][1]

	average = 0
	for j in range(6):
		mult = multiplier(re.sub("[0-9|.]", "", timelist[(index*2)+j]))
		average = average + (float(re.sub("[m|u|s]", "", timelist[(index*2) + j])) * mult)

	average = average/3

	print bench + '_' + size + ':'
	print str('{0:f}'.format(average)) + 'ms'

	index = index + 3'''
