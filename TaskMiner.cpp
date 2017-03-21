//LLVM IMPORTS
#include "llvm/Analysis/LoopInfo.h" 
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Attributes.h"

//MY IMPORTS
#include "TaskMiner.h"
#include "llvm/IR/Instructions.h"
#define DEBUG_TYPE "TaskMiner"
#define DEBUG_PRINT 

using namespace llvm;

STATISTIC(NumTasks, "Total Number of Tasks");
STATISTIC(NumFunctionCallTasks, "Number of FunctionCallTasks");
STATISTIC(NumIrregularLoops, "Number of irregular loops");

char TaskMiner::ID = 0;
static RegisterPass<TaskMiner> B("taskminer", "Task Miner: finds regions or \
	function calls inside loops that can be parallelized by tasks");


TaskMiner::~TaskMiner()
{
	// for (auto &t : tasks)
	// 	delete t;
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

	// getLoopsInfo(F);
	mineFunctionCallTasks(F);

	//Resolve each task's ins and outs sets
	resolveInsAndOutsSets();

	return false;
}

bool TaskMiner::doFinalization(Module &M)
{
	getStats();

	#ifdef DEBUG_PRINT
		int i = 0;
		for (auto &task : tasks)
		{
			errs() << "Task #" << i;
			task->print(errs());
			i++;
		}
		// printLoops();
	#endif

	return false;
}


void TaskMiner::printLoops()
{
	for (auto &loop : loops)
	{
		errs() << loop.first << ":\n";
		loop.second.print();
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
		if (l.second.indVar == nullptr)
			continue;
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

Loop* Task::getParent() { return parent; }
std::set<Value*> Task::getLiveIN() const { return liveIN; }
std::set<Value*> Task::getLiveOUT() const { return liveOUT; }
std::set<Value*> Task::getLiveINOUT() const { return liveINOUT; }

AccessType Task::getTypeFromInst(Instruction* I)
{
	if (dyn_cast<GetElementPtrInst>(I))
	{
		AccessType T = AccessType::UNKNOWN;
		for (auto user : I->users())
		{
			if (Instruction* inst = dyn_cast<Instruction>(user))
				T = T | getTypeFromInst(inst);
		}
		return T;
	}
	else if (dyn_cast<LoadInst>(I)) return AccessType::READ;
	else if (dyn_cast<StoreInst>(I)) return AccessType::WRITE;
	else return AccessType::UNKNOWN;
}

std::string Task::accessTypeToStr(AccessType T)
{
	switch(T)
	{
		case AccessType::READ:
			return "READ";
		case AccessType::WRITE:
		 	return "WRITE";
		case AccessType::READWRITE:
			return "READWRITE";
		default:
			return "UNKNOWN";
	}
}

raw_ostream& Task::printLiveSets(raw_ostream& os) const
{
	os << "\nIN:\n\t";
	for (auto &in : liveIN)
	{
		in->print(os);
		os << "\n";
	}
	os << "OUT:\n\t";
	for (auto &out : liveOUT)
	{
		out->print(os);
		os << "\n";
	}
	os << "INOUT:\n\t";
	for (auto &inout : liveINOUT)
	{
		inout->print(os);
		os << "\n";
	}

	return os;
}

raw_ostream& Task::print(raw_ostream& os) const
{
	return printLiveSets(os);
}

//CallInst* FunctionCallTask::getFunctionCall() { return functionCall; }

raw_ostream& FunctionCallTask::print(raw_ostream& os) const
{
	os << "\n===========\n" 
					<< "Type: Function Call Task"
					<< "\n===========\n"
					<< "Function Call: \n\t";
	functionCall->print(os);
	printLiveSets(os);
	return os;
}

bool FunctionCallTask::resolveInsAndOutsSets()
{
	Function* F = functionCall->getCalledFunction();
	std::map<Value*, AccessType> parameterAccessType;
	std::map<Value*, Value*> matchArgsParameters;
	Value* V;

	//Match parameters with arguments
	auto arg_aux = F->arg_begin();
	for (unsigned i = 0; i < functionCall->getNumArgOperands(); i++)
	{
		V = functionCall->getArgOperand(i);
		matchArgsParameters[arg_aux] = V;
		arg_aux++;
	}

	//Find the access types of the parameters
	// errs() << "Function " << F->getName() << ":\n";
	for (auto &arg : F->getArgumentList())
	{
		// arg.print(errs());
		errs() << "\n";
		if (!arg.user_empty())
			for (auto user : arg.users())
			{
				if (Instruction* I = dyn_cast<Instruction>(user))
				{
					AccessType T = getTypeFromInst(I);
					if (parameterAccessType.find(&arg) == parameterAccessType.end())
						parameterAccessType[&arg] = T;
					else
						parameterAccessType[&arg] = parameterAccessType[&arg] | T;
				}
			}
	}

	//Resolve ins and outs sets
	for (auto &p : parameterAccessType)
	{
		switch (p.second)
		{
			case AccessType::READ:
				liveIN.insert(matchArgsParameters[p.first]);
				break;
			case AccessType::WRITE:
				liveOUT.insert(matchArgsParameters[p.first]);
				break;
			case AccessType::READWRITE:
				liveINOUT.insert(matchArgsParameters[p.first]);
				break;
			case AccessType::UNKNOWN:
				liveIN.insert(matchArgsParameters[p.first]);
				break;
		}
	}

	return true;
}

CallInst* FunctionCallTask::getFunctionCall() const { return functionCall; }

