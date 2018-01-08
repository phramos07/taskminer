# SKELETON OF TASKMINER ALGORITHM
# 
# This script implements the TaskMiner infrastructure in a high level basis.
# Its purpose is a better understanding of the algorithm's flow and complexity.
# The algorithm receives a program P as input and returns a program P' annotated
# with OpenMP task based directives.
# 
# The algorithm has time complexity of O(N^2), in which N is the number of
# instructions in the input program P.

# MAIN PROCEDURE. It finds all potential tasks in a program P.
def minetasks(P):
	CFG = ControlFlowGraph(P) # O(N^2)
	PDG = ProgramDependenceGraph(P)	# O(N^2)
	H = findHelices(PDG);	# O(N^2)
	TASKREGIONS = []
	for h in H:	# O(N^2)
		reg = CFG.findMinimalCoveringRegion(h)
		TASKREGIONS.append(reg)
	for r in TASKREGIONS: # O(N^3)
		r.expand();

	ANN = Annotator()														
	NEW_PROGRAM = ANN.annotate(P, TASKREGIONS) # O(N^2)

	return NEW_PROGRAM

# =======================
# Below lie all the types abstractions used in TaskMiner.
# =======================

# Stub for Graph type
class Edge:
	def __init__(self, src, dst, kind):
		self.src = src
		self.dst = dst
		self.kind = kind

class Graph:
	def __init__(self):
		self.edges = []
		self.nodes = []

	def appendEdge(self, src, dst, kind):
		e = Edge(src, dst, kind)
		self.edges.append(e)
		self.nodes.append(src)
		self.nodes.append(dst)

	def visit():
		return

# The CFG is used to find the covering region, besides 
# Algorithm that finds the minimal covering region of a helix.
# Time Complexity: O(N) in terms of instructions.
class ControlFlowGraph(Graph):
	def __init__(self, P):
		Graph.__init__(self);
		self.PROGRAM = P

	def findMinimalCoveringRegion(helix):
		R = Region()
		f()
		return R

# This class is used to find SCC's and helices in the program.
# The algorithm that finds SCC's is the tarjan's algorithm, and it
# runs in O(N^2)
class ProgramDependenceGraph(Graph):
	def __init__(self, P):
		Graph.__init__(self)
		self.PROGRAM = P
		self.SCCs = []

	def getSCCs(self):
		if (not self.SCCs):
			self.SCCs = self.tarjanVisit()
		return self.SCCs

	def tarjanVisit(self):
		sccs = []
		f()
		return sccs

# This class will be used to store a task region.
# The main algorithm here is the expansion, which is O(N).
class Region:
	def __init__(self):
		self.basicblocks = []
		self.level = 0 # each region starts at a nest level 0. it's not absolute, it's a relative attribute
		self.dependencies = 0

	# Expansion of one level is O(N)
	def expand(self):
		canExpand = True
		while (canExpand):
			self.basicblocks.append(f()) # add new basic blocks to the list
			self.resolveDependencies()
			self.level += 1
			canExpand = self.checkExpansion()

	# it checks:
		# - if two tasks have merged O(N) 
		# - if dependencies have been conserved or reduced O(1)
		# - if parallelism has not decreased O(1)
	# it also does:
		# - privatization analysis O(N)
	def checkExpansion(self):
		return f()  

	def resolveDependencies(self):
		return f() # resolve dependencies for a given set of basic blocks

# I don'w know much about the annotator. 
# I know it receives a program and generates 
# a text to be applied to it. It probably runs in O(N^2)
class Annotator:
	def __init__(self): 
		self.text = ""
		f()

	def annotate(self, P, listOfRegions):
		self.P = P
		P_new = Program(f())
		return P_new

# Stub for Program type
class Program:
	def __init__(self, txt):
		self.source = txt

	def str(self):
		return self.source

# =======================
# Below lie the procedures invoked during the mining of tasks
# =======================

# It finds a helix for a given set of nodes. O(N)
def findHelix(scc):
	return f() 

# It finds all helices in the Program Dependence Graph. O(N^2)
def findHelices(PDG):
	helices = []
	for scc in PDG.getSCCs():
		helices.append(findHelix(scc))
	return helices

# Stub function for all methods that do any kind of heavy computation.
def f():
	return 1

if (__name__ == '__main__'):
	P = Program(input)
	annotated_P = minetasks(P) # O(N^2)