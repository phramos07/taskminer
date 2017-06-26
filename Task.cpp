#include "llvm/Analysis/LoopInfo.h" 
#include "llvm/IR/Instructions.h"

#include "Task.h"

using namespace llvm;

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

