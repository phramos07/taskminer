#include "llvm/Pass.h"
#include "llvm/ADT/Statistic.h"

//LOCAL IMPORTS
#include "DepAnalysis.h"
#include "RegionTree.h"
#include "Graph.hpp"
#include "Task.h"
#include "CostModel.h"

namespace llvm
{
	class TaskMiner : public ModulePass
	{
	private:
		//ModuleTaskGraph 
		RegionTree* taskGraph;
		DepAnalysis* DP=0;
		std::map<Function*, RegionTree*> RTs;	
		std::map<Function*, RegionTree*> getAllRegionTrees(Module &M);

		//tasks collected for this module
		std::list<Task*> tasks;

		//record of each instcall and their edges in the task graph
		std::map<Edge<RegionWrapper*, EdgeDepType>*, CallInst*> callInsts;

		//record of every scc in the taskgraph
		std::set<Graph<RegionWrapper*, EdgeDepType>* > SCCs;

		//keep alias sets

		//Keep functions that have been turned into tasks
		std::set<Function*> function_tasks;

	public:
		static char ID;
		TaskMiner() : ModulePass(ID) {}
		~TaskMiner() { /*delete taskGraph;*/ }
		void getAnalysisUsage(AnalysisUsage &AU) const override;
		bool runOnModule(Module &M) override;
		RegionTree* gettaskGraph(Module &M);
		void printRegionInfo();
		bool findRegionWrapperInSCC(RegionWrapper* RW);
		void mineRegionTasks();
		void mineRecursiveTasks();
		void mineFunctionCallTasks();
		void mineTasks();
		void resolveInsAndOutsSets();
		void computeCosts();
		void computeStats(Module &M);
		void computeTotalCost();

		std::list<CallInst*> getLastRecursiveCalls() const;

		//printing method
		void printTasks();

		//method to find all alias sets

	};
}