import re

f = open('acc.out', 'r')

string = f.read()

bench = re.compile('BENCH:(.*)  \|\|')
size = re.compile('SIZE:(.*)  ')
gpu = re.compile('GPU Runtime: (.*)s')
cpu = re.compile('CPU Runtime: (.*)s')

bench_list = bench.findall(string)
size_list = size.findall(string)
gpu_times = gpu.findall(string)
cpu_times = cpu.findall(string)

for i in range(len(bench_list)/2):
	x = i*8
	avg_cpu = 0
	avg_gpu = 0
	print bench_list[i] + '_' + size_list[i] + ' times:'
	for i in range(5):
		avg_gpu += float(gpu_times[x+i])
		avg_cpu += float(cpu_times[x+i])
	avg_gpu = (avg_gpu/5)
	avg_cpu = (avg_cpu/5)
	print 'GPU avg: ' + str('{0:f}'.format(avg_gpu))
	print 'CPU avg: ' + str('{0:f}'.format(avg_cpu)) + '\n'

print 'NOW FOR COALESCED TIMES:\n\n\n'

for i in range(len(bench_list)/2, len(bench_list)):
	x = i*8
	avg_cpu = 0
	avg_gpu = 0
	print bench_list[i] + '_' + size_list[i] + ' times:'
	for i in range(5):
		avg_gpu += float(gpu_times[x+i])
		avg_cpu += float(cpu_times[x+i])
	avg_gpu = (avg_gpu/5)
	avg_cpu = (avg_cpu/5)
	print 'GPU avg: ' + str('{0:f}'.format(avg_gpu))
	print 'CPU avg: ' + str('{0:f}'.format(avg_cpu)) + '\n'
