#include "LiveSets.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

char LiveSets::ID = 0;
static RegisterPass<LiveSets> W("livesets", 
	"Get the LiveSets and their access types inside a function");

void LiveSets::getAnalysisUsage(AnalysisUsage &AU) const
{
	AU.setPreservesAll();
}


bool LiveSets::runOnFunction(Function &func)
{
	for (auto &arg : func.getArgumentList())
	{
		// arg.print(errs());
		errs() << "\n";
		if (!arg.user_empty())
			for (auto user : arg.users())
			{
				if (Instruction* I = dyn_cast<Instruction>(user))
					addValue(&arg, getTypeFromInst(I));
			}
	}

	errs() << "Function " << func.getName() << ":\n";
	printLiveSets();

	return false;
}

void LiveSets::addValue(Value* V, AccessType T)
{
	if (args.find(V) == args.end())
		args[V] = T;
	else
		args[V] = args[V] | T;
}

AccessType LiveSets::getTypeFromInst(Instruction* I)
{
	if (dyn_cast<LoadInst>(I)) return AccessType::READ;
	else if (dyn_cast<StoreInst>(I)) return AccessType::WRITE;
	else return AccessType::UNKNOWN;
}

std::string LiveSets::accessTypeToStr(AccessType T)
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

void LiveSets::printLiveSets()
{
	for (auto &v : args)
	{
		v.first->print(errs());
		errs() 	
						<< " => "
						<< accessTypeToStr(v.second)
						<< "\n";
	}
}

