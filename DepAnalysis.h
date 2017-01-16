#ifndef DEP_ANALYSIS_H
#define DEP_ANALYSIS_H

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"

#include "ControlDependenceGraph.h"
#include "ProgramDependenceGraph.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/PostDominators.h"

#include <set>
#include <map>
#include <string>
#include <fstream>
#include <iomanip>

namespace llvm {
	class DepAnalysis : public FunctionPass
	{
	private:
		ProgramDependenceGraph* G;

	public:	
		static char ID;
		DepAnalysis() : FunctionPass(ID) {}
		~DepAnalysis() {};
		void getAnalysisUsage(AnalysisUsage &AU) const override;
		bool runOnFunction(Function &F) override;
		ProgramDependenceGraph* getDepGraph();

	};	
}



#endif

