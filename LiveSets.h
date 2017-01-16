#ifndef __LIVE_SETS_H__
#define __LIVE_SETS_H__

//LLVM IMPORTS
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"

//STL IMPORTS
#include <set>
#include <map>
#include <string>

namespace llvm
{
	enum AccessType {UNKNOWN=0, READ=1, WRITE=2, READWRITE=3};

	inline AccessType operator|(AccessType a, AccessType b)
	{
		return static_cast<AccessType>(static_cast<int>(a) | static_cast<int>(b)); 
	}

	typedef std::map<Value*, AccessType> ValueAccessType;

	class LiveSets : public FunctionPass
	{
	private:
		ValueAccessType args;
		void addValue(Value* V, AccessType T=AccessType::UNKNOWN);
		AccessType getTypeFromInst(Instruction* I);
		void matchFormalParametersWithArguments();

	public:
		static char ID;
		LiveSets() : FunctionPass(ID) {}
		~LiveSets() {};
		void getAnalysisUsage(AnalysisUsage &AU) const override;
		bool runOnFunction(Function &func) override;

		//Only for debugging purposes
		std::string accessTypeToStr(AccessType T);
		void printLiveSets();


	};
}

#endif