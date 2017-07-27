#include "llvm/Analysis/LoopInfo.h" 
#include "llvm/IR/Instructions.h"

#include "Task.h"

using namespace llvm;

std::set<Value*> Task::getLiveIN() const { return liveIN; }
std::set<Value*> Task::getLiveOUT() const { return liveOUT; }
std::set<Value*> Task::getLiveINOUT() const { return liveINOUT; }
std::set<BasicBlock*> Task::getbbs() const { return bbs; }

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
		case AccessType::UNKNOWN:
			return "UNKNOWN";
		default:
			llvm_unreachable("T should be one of the 4 access types: 0-UNK, 1-REA, 2-WRT, 3-R/W");
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

FunctionCallTask::FunctionCallTask(CallInst* CI)
	: Task(FCALL_TASK)
	, functionCall(CI)
	{
		Function* F = functionCall->getCalledFunction();
		for (Function::iterator BB = F->begin(); BB != F->end(); ++BB)
		{
			bbs.insert(BB);
		}
	}

raw_ostream& FunctionCallTask::print(raw_ostream& os) const
{
	os << "\n===========\n" 
					<< "Type: Function Call Task"
					<< "\n===========\n"
					<< "Function Call: \n\t";
	functionCall->print(os);
	printLiveSets(os);
	CM.print(os);

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

CostModel FunctionCallTask::computeCost()
{
	uint32_t n_insts = 0;
	uint32_t n_indeps = liveIN.size();
	uint32_t n_outdeps = liveOUT.size();

	Function* F = functionCall->getCalledFunction();
	std::stack<Function*> funcs;
	funcs.push(F);
	while (!funcs.empty())
	{
		Function* func = funcs.top();
		funcs.pop();
		if (func->empty())
			continue;
		for (Function::iterator bb = func->begin(); bb != func->end(); ++bb)
			for (BasicBlock::iterator it = bb->begin(); it != bb->end(); ++it)
			{
				n_insts++;
				if (CallInst* CI = dyn_cast<CallInst>(it))
				{
					funcs.push(CI->getCalledFunction());
				}
			}
	}

	n_insts = n_insts*10;

	CM.setData(n_insts, n_indeps, n_outdeps);

	return CM;
}


RecursiveTask::RecursiveTask(CallInst *CI, bool isInsideLoop)
	: recursiveCall(CI)
	, isInsideLoop(isInsideLoop)
	, Task(RECURSIVE_TASK)
	{
		Function* F = recursiveCall->getCalledFunction();
		for (Function::iterator BB = F->begin(); BB != F->end(); ++BB)
		{
			bbs.insert(BB);
		}		
	}

CallInst* RecursiveTask::getRecursiveCall() const
{
	return recursiveCall;
}

bool RecursiveTask::resolveInsAndOutsSets()
{
	return true;
}

raw_ostream& RecursiveTask::print(raw_ostream& os) const
{
	os << "\n===========\n" 
					<< "Type: Recursive Call Task"
					<< "\n===========\n"
					<< "Recursive Call: \n\t";
	recursiveCall->print(os);
	printLiveSets(os);
	CM.print(os);

	return os;	
}

CostModel RecursiveTask::computeCost()
{
	uint32_t n_insts = 0;
	uint32_t n_indeps = liveIN.size();
	uint32_t n_outdeps = liveOUT.size();

	Function* func = recursiveCall->getCalledFunction();
	for (Function::iterator bb = func->begin(); bb != func->end(); ++bb)
			for (BasicBlock::iterator it = bb->begin(); it != bb->end(); ++it)
				n_insts++;

	n_insts = n_insts * 10;

	CM.setData(n_insts, n_indeps, n_outdeps);

	return CM;
}

bool RegionTask::resolveInsAndOutsSets()
{
	return true;
}

raw_ostream& RegionTask::print(raw_ostream& os) const
{
	os << "\n===========\n" 
					<< "Type: Region Task"
					<< "\n===========\n"
					<< "BBs: \n\t";
	for (auto bb : bbs)
		os << " " << bb->getName();
	printLiveSets(os);
	CM.print(os);

	return os;	
}

CostModel RegionTask::computeCost()
{
	uint32_t n_insts = 0;
	uint32_t n_indeps = liveIN.size();
	uint32_t n_outdeps = liveOUT.size();

	for (auto bb : bbs)
			for (BasicBlock::iterator it = bb->begin(); it != bb->end(); ++it)
				n_insts++;

	n_insts = n_insts * 10;

	CM.setData(n_insts, n_indeps, n_outdeps);

	return CM;
}