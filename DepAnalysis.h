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
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/PostDominators.h"

#include <set>
#include <list>
#include <map>
#include <string>
#include <fstream>
#include <iomanip>

namespace llvm {

	struct Windmill;
	struct Helix;
	struct LoopData;

	class DepAnalysis : public FunctionPass
	{
	private:
		ProgramDependenceGraph* G;
		std::set<std::set<GraphNode*> > SCCs;
		void createProgramDependenceGraph(Function &F);
		std::map<Loop*, LoopData*> loops;
		LoopInfo *LI;
		RegionInfo *RI;


	public:	
		static char ID;
		static int numberOfLoopData;
		DepAnalysis() : FunctionPass(ID) {}
		~DepAnalysis() {};
		void getAnalysisUsage(AnalysisUsage &AU) const override;
		bool runOnFunction(Function &F) override;
		ProgramDependenceGraph* getDepGraph();
		void getLoopsInfo(Function &F);
		void findWindmills();
		void findHelices();
		Region *getMinimumCoverRegion(std::set<Instruction*> insts);
		void findMinimumRegionForEachHelix();

		//printing methods
		void dumpWindmillsToDot(Function &F, Windmill &W, int windmillId); 
		void dumpWindmillsToDot(Function &F);
		raw_ostream& print(raw_ostream& os=errs()) const;

	};

	struct Windmill
	{
		Windmill() {}
		~Windmill() { delete H; }
		std::set<GraphNode*> nodes;
		Helix* H;
		
		raw_ostream& print(raw_ostream& os=errs()) const;
	};

	struct Helix
	{
		Helix() {}
		~Helix() {} 
		std::set<GraphNode*> subgraph;
		bool hasSCC;

		raw_ostream& print(raw_ostream& os=errs()) const;
	};

	struct LoopData
	{
		LoopData() : regular(false) {}
		LoopData(bool reg) : regular(reg) {}
		~LoopData() { delete W; indVar=0; }
		Windmill* W=0;
		Instruction* indVar=0;
		bool regular;
		int id;
		Region *R=0;

		//Debugging purposes only

		raw_ostream& print(raw_ostream& os=errs()) const;
	};	
}





#endif

