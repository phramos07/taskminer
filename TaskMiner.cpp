//LLVM IMPORTS
#include "llvm/Analysis/LoopInfo.h" 
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Attributes.h"

//MY IMPORTS
#include "TaskMiner.h"
#include "llvm/IR/Instructions.h"
<<<<<<< HEAD
#define DEBUG_TYPE "print-tasks"
#define DEBUG_TYPE "print-loop"
=======
#define DEBUG_TYPE "TaskMiner"
//define DEBUG_PRINT 
>>>>>>> b50642b2cf609c4bffc2dc6cd485e8de4f87ddac

using namespace llvm;

STATISTIC(NumTasks, "Total Number of Tasks");
STATISTIC(NumFunctionCallTasks, "Number of FunctionCallTasks");
STATISTIC(NumIrregularLoops, "Number of irregular loops");

char TaskMiner::ID = 0;
static RegisterPass<TaskMiner> B("taskminer", "Task Miner: finds regions or \
	function calls inside loops that can be parallelized by tasks");


TaskMiner::~TaskMiner()
{
	for (auto &t : tasks)
		delete t;
}

void TaskMiner::getAnalysisUsage(AnalysisUsage &AU) const
{
	AU.addRequired<LoopInfoWrapperPass>();
	// AU.addRequired<DepAnalysis>();
	AU.setPreservesAll();
}

bool TaskMiner::runOnFunction(Function &F)
{
	if (F.isDiscardableIfUnused())
		return false;

	LIWP = &(getAnalysis<LoopInfoWrapperPass>());
	LI = &(LIWP->getLoopInfo());
	// DA = &(getAnalysis<DepAnalysis>());

	//DEPENDENCE ANALYSIS MUST RETURN WINDMILLS WITH HELICES


	getLoopsInfo(F); //should be in depanalysis
	mineFunctionCallTasks(F); //should be more general

	//Resolve each task's ins and outs sets for every task
	resolveInsAndOutsSets();

	return false;
}

bool TaskMiner::doFinalization(Module &M)
{
	getStats();

<<<<<<< HEAD
	int i = 0;
	for (auto &task : tasks)
	{

		DEBUG_WITH_TYPE("print-tasks", errs() << "Task #" << i);
		DEBUG_WITH_TYPE("print-tasks", task->print(errs()));
		i++;
	}
	DEBUG(printLoops());
=======
	#ifdef DEBUG_PRINT
		int i = 0;
		for (auto &task : tasks)
		{
			//errs() << "Task #" << i;
			//task->print(errs());
			i++;
		}
		// printLoops();
	#endif
>>>>>>> b50642b2cf609c4bffc2dc6cd485e8de4f87ddac

	return false;
}


void TaskMiner::printLoops()
{
	for (auto &loop : loops)
	{
//		errs() << loop.first << ":\n";
//		loop.second.print();
		errs() << "\n";
	}
}

void TaskMiner::LoopData::print()
{
	errs()	<< "INDVAR: " 
					<< indVar->getName()
					<< "\nBASICBLOCKS: ";
	for (std::list<BasicBlock*>::iterator it = innerBBs.begin(); 
		it != innerBBs.end(); ++it)
	{
		errs () << (*it)->getName() << " | ";
	}
	errs() << "\n";
}

void TaskMiner::resolveInsAndOutsSets()
{
	for (auto &t : tasks)
		t->resolveInsAndOutsSets();
}

void TaskMiner::getStats()
{
	for (auto &t : tasks)
	{
		NumTasks++;
		if (FunctionCallTask* FCT = dyn_cast<FunctionCallTask>(t))
			NumFunctionCallTasks++;
	}
}

void TaskMiner::getLoopsInfo(Function &F)
{
	//Find all Loops
	for (Function::iterator BB = F.begin(); BB != F.end(); ++BB)
	{
		//If BB is in loop, then we can try to get the induction var from this loop
		Loop* loop = LI->getLoopFor(BB);
		if (loop)
		{
			if (loops.find(loop) == loops.end()) //If the loop hasn't been added to the loopdata yet
			{
				TaskMiner::LoopData LD;
				loops[loop] = LD;
				loops[loop].innerBBs.push_back(BB);
				Instruction* inst = loop->getCanonicalInductionVariable();
				loops[loop].indVar = inst;
			}
			else
			{
				loops[loop].innerBBs.push_back(BB);
			}
		}
	}		
	//colaesce loops by induction variable
}

void TaskMiner::mineFunctionCallTasks(Function &F)
{
	//If LOOP is irregular, finds the region inside the loop
	//TODO: DEPANALYSIS
	//HYPOTHESIS ZERO: ALL LOOPS ARE IRREGULAR/TASK-PARALLELIZABLE

	//Now go through the loops and extract which sort of task it is. for now
	//we don't have any heuristics to decide whether it will be a code fragment
	//task or a mere function call task. So we're focused only on function calls.
	Function* calledF;
	std::set<Instruction*> fCalls;
	for (auto &l : loops)
	{
		// l.second.print();
		// if (l.second.indVar == nullptr)
		// 	continue;
		for (auto &bb : l.second.innerBBs)
		{
			for (BasicBlock::iterator func = bb->begin(); func != bb->end(); ++func)
			{
				if (CallInst* CI = dyn_cast<CallInst>(func))
				{
					calledF = CI->getCalledFunction();
					if ((CI->getModule() == F.getParent()) 
								&& (!calledF->isDeclaration())
								&& (!calledF->isIntrinsic())
								&& (fCalls.find(CI) == fCalls.end())
								&& (!CI->getDereferenceableBytes(0)))
							{
								Task* functionCallTask = new FunctionCallTask(l.first, CI);
								insertNewFunctionCallTask(*functionCallTask);
								fCalls.insert(CI);
							}
				}
			}
		}
	}
}

void TaskMiner::insertNewFunctionCallTask(Task &T)
{
	if (FunctionCallTask* FCT = dyn_cast<FunctionCallTask>(&T))
	{
		for (auto &t : tasks)
		{
			if (FunctionCallTask* FCT_cmp = dyn_cast<FunctionCallTask>(t))
				if (FCT_cmp->getFunctionCall() == FCT->getFunctionCall())
					return;
		}

		tasks.push_back(&T);
	}
}

std::list<Task*> TaskMiner::getTasks()
{
	return tasks;
}