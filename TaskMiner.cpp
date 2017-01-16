//LLVM IMPORTS
#include "llvm/Analysis/LoopInfo.h" 
//#include "llvm/Analysis/Dominators.h"

//MY IMPORTS
#include "TaskMiner.h"
#define DEBUG_TYPE "TaskMiner"
// #define DEBUG_ 

using namespace llvm;

void TaskMiner::getAnalysisUsage(AnalysisUsage &AU) const
{
	AU.addRequired<DepAnalysis>();
	AU.addRequired<LiveSets>();
	AU.setPreservesAll();
}

bool TaskMiner::runOnFunction(Function &func)
{

	return false;
}



char TaskMiner::ID = 0;
static RegisterPass<TaskMiner> B("taskminer", "Task Miner: finds regions or \
	function calls inside loops that can be parallelized by tasks");

