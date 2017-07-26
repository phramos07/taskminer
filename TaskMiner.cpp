//llvm imports
#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Support/CommandLine.h"

//local imports
#include "TaskMiner.h"
#include "DepAnalysis.h"
#define DEBUG_TYPE "print-tasks"

using namespace llvm;

char TaskMiner::ID = 0;
static RegisterPass<TaskMiner> E("taskminer", "Run the TaskMiner algorithm on a Module", false, false);

static cl::opt<bool, false> printTaskGraph("print-task-graph",
  cl::desc("Print dot file containing the TASKGRAPH"), cl::NotHidden);

STATISTIC(NTASKS, "Total number of tasks");
STATISTIC(NFCALLTASKS, "Total number of function call (non-recursive) tasks");
STATISTIC(NRECURSIVETASKS, "Total number of recursive tasks");
STATISTIC(NREGIONTASKS, "Total number of region tasks");

void TaskMiner::getAnalysisUsage(AnalysisUsage &AU) const
{
	AU.addRequired<DepAnalysis>();
	AU.addRequired<RegionInfoPass>();
	AU.setPreservesAll();
}

bool TaskMiner::runOnModule(Module &M)
{
	//STEP 1: GET TASK GRAPH FOR THE WHOLE MODULE
	taskGraph = gettaskGraph(M);

	if (printTaskGraph)
	{
		taskGraph->dumpToDot(M.getName(), true);
	}

	//STEP2: FIND SCC'S IN THE TASKGRAPH.
	SCCs = taskGraph->getStronglyConnectedSubgraphs();

	//STEP3: MINE FOR TASKS
	mineTasks();
	
	//STEP4: FIND THE INS/OUTS OF EACH TASK. INCLUDING ALIASING.
	resolveInsAndOutsSets();

	//STEP5: COMPUTE THE COSTS OF EACH TASK.

	//STEP6: GIVE IT OUT TO THE ANNOTATOR.

	DEBUG_WITH_TYPE("print-tasks", printRegionInfo());
	DEBUG_WITH_TYPE("print-tasks", printTasks());

	return false;
}

std::map<Function*, RegionTree*> TaskMiner::getAllRegionTrees(Module &M)
{
	std::map<Function*, RegionTree*> RTs;

	for (Module::iterator F = M.begin(); F != M.end(); ++F)
	{
		if (!F->empty())
		{	
			DepAnalysis *DP = &(getAnalysis<DepAnalysis>(*F));
			RTs[F] = std::move((DP->getRegionTree()));
		}
	}

	return RTs;
}

void TaskMiner::printRegionInfo()
{
	// for (auto rt : RTs)
	// {
	// 	errs() << "\n\nREGIONTREE FOR FUNCTION: " << rt.first->getName();
	// 	rt.second->print(errs());
	// }

	errs() << "\n\nWHOLE REGION GRAPH:";
	if (taskGraph)
		taskGraph->print(errs());
}

RegionTree* TaskMiner::gettaskGraph(Module &M)
{
	//0: Get all region graphs for each function.
	taskGraph = new RegionTree();
	if (RTs.empty())
		RTs = getAllRegionTrees(M);

	//1: Merge all the region graphs into a single one 
	//that will be our task graph
	auto it = RTs.begin();
	for (auto rt = RTs.begin(); rt != RTs.end(); rt++)
	{
		taskGraph->mergeWith(rt->second);
	}

	//2: GET ALL RECURSIVE EDGES AND COPY A HUB FOR EACH.
	//WE'LL HAVE ALL THE RT'S HUBS AND THEIR DESIGNED EDGES->SRC().
	//FOR EACH EDGE->SRC() WE'LL CONNECT THEM TO THE TOP LEVEL OF THE HUB (GET TOP LEVEL() REGION TREE METHOD)
	//THAT'S IT. 
	std::map<Edge<RegionWrapper*, EdgeDepType>*, RegionTree*> recToHub;
	for (auto e : taskGraph->getEdges())
	{
		if (e->getType() == EdgeDepType::RECURSIVE)
		{
			Function *F = e->getDst()->getItem()->F;
			auto recHub = taskGraph->copyRegionTree(RTs[F]);
			recToHub[e] = recHub;
		}
	}

	//Remove recursive edges before merging ? (maybe)

	//3: Now we merge each hub into the taskGraph
	for (auto pair : recToHub)
	{
		taskGraph->mergeWith(pair.second);
	}

	//Now add fcall edges

	//THIS SHOULD BE THE LAST THING:
	//4: Go through every instruction to create the FCALL edges.
	//only do it if FCALL != FUNCTION, don't do that for recursive cases.
	//ALWAYS CONNECT TO THE REAL TOPLEVEL. I'll need to connect real -> toplevel and hub -> toplevel
	for (Module::iterator F = M.begin(); F != M.end(); ++F)
	{
		if (F->empty())
			continue;
		RegionInfoPass* RIP = &(getAnalysis<RegionInfoPass>(*F));
		RegionInfo* RI = &(RIP->getRegionInfo());
		for (Function::iterator BB = F->begin(); BB != F->end(); ++BB)
		{
			for (BasicBlock::iterator I = BB->begin(); I != BB->end(); ++I)
			{
				if (CallInst* CI = dyn_cast<CallInst>(I))
				{
					Function* calledF = CI->getCalledFunction();
					if ((calledF != F) && (!calledF->empty()))
					{
						Region* R = RI->getRegionFor(CI->getParent());
						Node<RegionWrapper*>* src = 
							taskGraph->getNode(R->getEntry(), R->getExit(), R->isTopLevelRegion());
						Node<RegionWrapper*>* dst = 
							taskGraph->getTopLevelRegionNode(calledF);

						auto e_ =	taskGraph->addEdge(src, dst, EdgeDepType::FCALL);
						//add to the map Edge -> CallInst
						callInsts[e_] = CI;
					}					
				}
			}
		}
	}

	//And add Fcall for recursive hubs
	for (auto pair : recToHub)
	{
		auto F = pair.first->getSrc()->getItem()->F;
		auto dst = pair.second->getTopLevelRegionNode(F, true);
		auto dstHub = taskGraph->getNode(dst->getItem());
		if (dstHub)
		{
			taskGraph->addEdge(pair.first->getSrc(), dstHub, EdgeDepType::FCALL);
		}
	}

	return taskGraph;
}

void	TaskMiner::mineFunctionCallTasks()
{
	//For each FCALL edge, check if
	// A) it comes from an SCC
	// B) it goes to an SCC
	// C) src and dst are different functions
	for (auto e : taskGraph->getEdges())
	{
		auto type = e->getType();
		if (type == EdgeDepType::FCALL)
		{
			auto srcRW = e->getSrc()->getItem();
			auto dstRW = e->getDst()->getItem();
			if (taskGraph->nodeReachesSCC(dstRW))
				errs() << "IT REACHES OMG";

			if ((srcRW->F != dstRW->F)
				&& ((srcRW->hasLoop))
				/*&& (taskGraph->nodeReachesSCC(dstRW))*/)
			{
				CallInst* CI = callInsts[e];
				Task* TASK = new FunctionCallTask(CI);
				tasks.push_back(TASK);
				NTASKS++;
				NFCALLTASKS++;
			}
		}
	}
}

bool TaskMiner::findRegionWrapperInSCC(RegionWrapper* RW)
{
	for (auto scc : SCCs)
	{
		if (scc->getNodeIndex(RW) != -1)
			return true;
	}

	return false;
}

void TaskMiner::mineTasks()
{
	mineFunctionCallTasks();
	// mineRegionTasks();
	// mineRecursiveTasks();
}

std::list<CallInst*> TaskMiner::getLastRecursiveCalls() const
{
	std::list<CallInst*> CIs;
	for (auto task : tasks)
	{
		if (RecursiveTask* RT = dyn_cast<RecursiveTask>(task))
		{
			if (RT->getPrev() && !RT->getNext())
			{
				CIs.push_back(RT->getRecursiveCall());
			}
		}
	}

	return CIs;
}

void TaskMiner::resolveInsAndOutsSets()
{
	for (auto task : tasks)
		task->resolveInsAndOutsSets();
}

void TaskMiner::printTasks()
{
	for (auto task : tasks)
	{
		task->print(errs());
	}
}
