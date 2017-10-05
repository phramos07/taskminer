//llvm imports
#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Support/CommandLine.h"

//local imports
#include "TaskMiner.h"
#include "Task.h"
#include "DepAnalysis.h"
#define DEBUG_TYPE "print-tasks"

using namespace llvm;

char TaskMiner::ID = 0;
static RegisterPass<TaskMiner> E("taskminer", "Run the TaskMiner algorithm on a Module", false, false);

static cl::opt<bool, false> printTaskGraph("print-task-graph",
  cl::desc("Print dot file containing the TASKGRAPH"), cl::NotHidden);

static cl::opt<bool, true> MINE_FCALLTASK("MINE_FCALLTASK",
  cl::desc("Enable the mining of Function Call Tasks"), cl::Hidden);

static cl::opt<bool, true> MINE_RECTASKS("MINE_RECTASKS",
  cl::desc("Enable the mining of Recursive Tasks"), cl::Hidden);

static cl::opt<bool, true> MINE_REGIONTASKS("MINE_REGIONTASKS",
  cl::desc("Enable the mining of Region Tasks"), cl::Hidden);

static cl::opt<bool, true> MINE_LOOPTASKS("MINE_LOOPTASKS",
  cl::desc("Enable the mining of Loop Tasks"), cl::Hidden);

static cl::opt<int> NUMBER_OF_THREADS("N_THREADS",
  cl::desc("Number of threads in the runtime"), cl::NotHidden);

static cl::opt<int> RUNTIME_COST("RUNTIME_COST",
  cl::desc("Minimum cost per task (in instructions) in the runtime"), cl::NotHidden);

STATISTIC(NTASKS, "Total number of tasks");
STATISTIC(NFCALLTASKS, "Total number of function call (non-recursive) tasks");
STATISTIC(NRECURSIVETASKS, "Total number of recursive tasks");
STATISTIC(NREGIONTASKS, "Total number of region tasks");
STATISTIC(NINSTSTASKS, "Total number of task instruction costs");
STATISTIC(NINDEPS, "Total number of task input dep costs");
STATISTIC(NOUTDEPS, "Total number of task output dep costs");
STATISTIC(NMODULEINSTS, "Total number of module instructions");


void TaskMiner::getAnalysisUsage(AnalysisUsage &AU) const
{
	AU.addRequired<DepAnalysis>();
	AU.addRequired<RegionInfoPass>();
	AU.addRequired<LoopInfoWrapperPass>();
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
	computeCosts();

	//STEP6: GIVE IT OUT TO THE ANNOTATOR.

	DEBUG_WITH_TYPE("print-tasks", printRegionInfo());
	DEBUG_WITH_TYPE("print-tasks", printTasks());

	computeStats(M);

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
	bool indirectRecursion;
	for (auto e : taskGraph->getEdges())
	{
		auto type = e->getType();
		if (type == EdgeDepType::FCALL)
		{
			auto srcRW = e->getSrc()->getItem();
			auto dstRW = e->getDst()->getItem();
			indirectRecursion=false;
			//Check if they're indirect recursion 
			//(if both src and dst are in the same SCC)
			for (auto scc : SCCs)
			{
				if ((scc->getNodeIndex(srcRW) != -1)
					&& (scc->getNodeIndex(dstRW) != -1))
				{
					indirectRecursion=true;
					break;
				}
			}
			CallInst* CI = callInsts[e];
			if (srcRW->hasLoop)
			{
				errs() << "mingin function call inside loop: ";
				CI->dump();
			}
			if ((srcRW->F != dstRW->F)
				/*&& ((srcRW->hasLoop))*/
				&& (!indirectRecursion)
				/*&& (taskGraph->nodeReachesSCC(dstRW))*/)
			{
				Task* TASK = new FunctionCallTask(CI);
				tasks.push_back(TASK);
				NTASKS++;
				NFCALLTASKS++;
				function_tasks.insert(srcRW->F);
			}
		}
	}
}

void TaskMiner::mineRecursiveTasks()
{
	//For mining recursive tasks, check
	//A) If edgetype is FCALL and srcF == dstF
	//B) Go through every Fcall inside F that matches F
	//C) Create a task for each and set prev/next
	std::list<Function*> rec_funcs;

	for (auto e : taskGraph->getEdges())
	{
		auto type = e->getType();
		if (type == EdgeDepType::FCALL)
		{
			auto srcRW = e->getSrc()->getItem();
			auto dstRW = e->getDst()->getItem();
			if (srcRW->F == dstRW->F)
			{
				//add to a list of Functions;
				rec_funcs.push_back(srcRW->F);
			}
		}
	}


	std::map<Function*, std::set<CallInst*> > rec_calls;

	//Go through the list of Functions
	for (auto F : rec_funcs)
	{
		for (Function::iterator bb = F->begin(); bb != F->end(); ++bb)
		{
			for (BasicBlock::iterator I = bb->begin(); I != bb->end(); ++I)
			{
				if (CallInst* CI = dyn_cast<CallInst>(I))
				{
					Function* calledF = CI->getCalledFunction();
					if (calledF == F)
					{
						rec_calls[F].insert(CI);
					}
				}
			}
		}
	}

	bool isInsideLoop;
	RecursiveTask* prev;
	Function* func;
	std::map<Function*, std::list<RecursiveTask*> > rec_tasks;

	//Create task for each recursive call
	for (auto pair : rec_calls)
	{
		prev = nullptr;
		isInsideLoop = false;
		func = pair.first;
		LoopInfoWrapperPass *LIWP = &(getAnalysis<LoopInfoWrapperPass>(*func));
		LoopInfo* LI = &(LIWP->getLoopInfo());
		auto bb = (*pair.second.begin())->getParent();
		if (LI->getLoopFor(bb)) //the fcall is inside loop
		{
			isInsideLoop = true;
		}
		for (auto CI = pair.second.begin(); CI != pair.second.end(); ++CI)
		{
			function_tasks.insert(func);
			RecursiveTask* TASK = new RecursiveTask(*CI, isInsideLoop);
			TASK->setPrev(prev);
			if (prev)
				prev->setNext(TASK);
			prev = TASK;
			rec_tasks[pair.first].push_back(TASK);
			

			tasks.push_back((Task*)TASK);
			NTASKS++;
			NRECURSIVETASKS++;
		}

	//Now what do we do here? We go through every recursive task in each function.
	//we call the region analysis. If the recursive calls are in different regions,
	//we don't add them to the main list of tasks.
	}
}

void TaskMiner::mineRegionTasks()
{
	//1: go through every SCC >= 2
	//2: check the function in every SCC. only analyse those which
	//haven't been added as a functioncall task or recursive task
	//3: Find the largest region that covers that SCC
	//4: Build RegionTask with those BB's that are within the largest region
	std::set<Graph<RegionWrapper*, EdgeDepType>* > candidate_sccs;

	for (auto scc : SCCs)
	{
		if (scc->size() > 1)
		{
			for (auto n : scc->getNodes())
			{
				if (std::find(function_tasks.begin(), 
					function_tasks.end(), n->getItem()->F) == function_tasks.end())
				{
					candidate_sccs.insert(scc);
				}
			}
		}
	}

	//For each candidate SCC, check if they're SCC's originated
	//from indirect recursion
	std::list<std::set<Graph<RegionWrapper*, EdgeDepType>* >::iterator> to_be_removed;
	for (std::set<Graph<RegionWrapper*, EdgeDepType>* >::iterator 
		scc = candidate_sccs.begin(); scc != candidate_sccs.end(); ++scc)
		for (auto e : (*scc)->getEdges())
		{
			if (e->getType() == EdgeDepType::FCALL)
			{
				to_be_removed.push_back(scc);
				break;
			}
		}

	//remove these scc's from indirect recursion
	for (auto scc : to_be_removed)
		candidate_sccs.erase(scc);

	//Now for every candidate scc,
	//1: find the largest region that covers all the nodes
	//2: create regiontask
	//3: add BB's to it
	for (auto scc : candidate_sccs)
	{
		auto firstRW = (*scc->getNodes().begin())->getItem();
		Function* F = firstRW->F;
		RegionInfoPass *RIP = &(getAnalysis<RegionInfoPass>(*F));
		RegionInfo* RI = &RIP->getRegionInfo();
		Region *R = RI->getRegionFor(firstRW->entry);
		for (auto node : scc->getNodes())
		{
			auto currentRW = node->getItem();
			Region* currentRegion = RI->getRegionFor(currentRW->entry);
			R = RI->getCommonRegion(R, currentRegion);
		}

		RegionTask* TASK = new RegionTask();

		//Now with the region R in hands, let's build the regiontask!
		//TODO: REMOVE ALL THE BASIC BLOCKS TAHT BELONG TO LOOP HEADER
		// THAT IS, ADD ONLY BASIC BLOCKS THAT ARE PART OF THE TASK (INSIDE)
		// THE OUTER MOST LOOP
		for (Function::iterator BB = F->begin(); BB != F->end(); ++BB)
		{
			if (R->contains(BB) && RI->getRegionFor(BB) != R)
			{
				Region *R_ = RI->getRegionFor(BB);
				TASK->addBasicBlock(BB);
				TASK->addBasicBlock(R_->getExit());
			}
		}

		tasks.push_back(TASK);
		NTASKS++;
		NREGIONTASKS++;
	}
}

void TaskMiner::mineLoopTasks()
{

}


void TaskMiner::mineTasks()
{
	mineRecursiveTasks();
	// mineFunctionCallTasks();
	// if (MINE_REGIONTASKS) mineRegionTasks();
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
	//Set static values: runtime cost and number of threads
	// CostModel::setNWorkers(NUMBER_OF_THREADS);
	// CostModel::setRuntimeCost(RUNTIME_COST);

	for (auto task : tasks)
		task->resolveInsAndOutsSets();
}

void TaskMiner::computeCosts()
{
	for (auto task : tasks)
		task->computeCost();
}

void TaskMiner::computeTotalCost()
{
	NOUTDEPS = 0;
	NINDEPS = 0;
	NINSTSTASKS = 0;

	for (auto task : tasks)
	{
		auto cost = task->getCost();
		NINSTSTASKS += cost.getNInsts();
		NINDEPS += cost.getNInDeps();
		NOUTDEPS += cost.getNOutDeps();
	}
}

void TaskMiner::computeStats(Module &M)
{
	//Compute total cost
	computeTotalCost();

	//Compute total number of instructions
	for (Module::iterator F = M.begin(); F != M.end(); ++F)
	{
		if (F->empty())
			continue;
		for (Function::iterator BB = F->begin(); BB != F->end(); ++BB)
			for (BasicBlock::iterator I = BB->begin(); I != BB->end(); ++I)
			{
				NMODULEINSTS++;
			}
	}
}

void TaskMiner::printTasks()
{
	for (auto task : tasks)
	{
		task->print(errs());
	}
}
