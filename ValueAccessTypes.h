#ifndef VALUE_ACCESS_TYPES_H
#define VALUE_ACCESS_TYPES_H

//LLVM IMPORTS
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Analysis/RegionInfo.h"

//LOCAL IMPORTS
#include "PtrRangeAnalysis.h"

namespace llvm{

	class ValueAccessTypes : public FunctionPass {

	public:
		enum AccessType{LOAD, STORE, LOADSTORE};
		static char ID;
		ValueAccessTypes() : FunctionPass(ID) {};
		~ValueAccessTypes() {};
	  virtual bool runOnFunction(Function &F) override;
	  virtual void getAnalysisUsage(AnalysisUsage &AU) const override;
	  std::map<Value*, ValueAccessTypes::AccessType> getValuesAcessTypes();
	
	private:
		std::map<Value*, ValueAccessTypes::AccessType> accessTypes;

	};

}

#endif