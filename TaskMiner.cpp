//llvm imports
#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Analysis/CallGraph.h"

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
	AU.addRequired<CallGraphWrapperPass>();
	AU.addRequired<RegionInfoPass>();
	AU.addRequired<LoopInfoWrapperPass>();
	AU.setPreservesAll();
}

bool TaskMiner::runOnModule(Module &M)
{
	errs() << "++++++++++++++++++++++++\n";
	errs() << "\tTASKMINER\n";
	errs() << "++++++++++++++++++++++++\n";

	//STEP 1: GET TASK GRAPH FOR THE WHOLE MODULE
	errs() << "\nSTEP 1: Generating Task Graph\n";
	taskGraph = gettaskGraph(M);

	if (printTaskGraph)
		taskGraph->dumpToDot(M.getName(), true);

	//STEP2: FIND SCC'S IN THE TASKGRAPH.
	errs() << "STEP 2: Finding SCC's in the Task Graph.\n";
	SCCs = taskGraph->getStronglyConnectedSubgraphs();

	//STEP3: MINE FOR TASKS - ITERATIVE EXPANSION
	errs() << "STEP 3: Mining Tasks\n";
	mineTasks(M);
	
	//STEP4: FIND THE INS/OUTS OF EACH TASK. INCLUDING ALIASING.
	errs() << "STEP 4: Determining dependences and memory regions.\n";
	resolveInsAndOutsSets();

	//STEP5: PRIVATIZATION
	errs() << "STEP 5: Privatization of variables.\n";
	resolvePrivateValues();

	//STEP6: COMPUTE THE COSTS OF EACH TASK.
	errs() << "STEP 6: Estimating the cost for region tasks.\n";
	computeCosts();

	//STEP7: GIVE IT OUT TO THE ANNOTATOR.
	errs() << "STEP 7: Infering Source Code information to Annotate the Tasks.\n";

	// DEBUG_WITH_TYPE("print-tasks", printRegionInfo());
	DEBUG_WITH_TYPE("print-tasks", printTasks());

	computeStats(M);

	return false;
}

//Determines the top level callsites for every recursive task.
void TaskMiner::determineTopLevelRecursiveCalls(Module &M)
{
	errs() << "\t\t\tDetermining top level recursive calls..\n";
	CallGraphWrapperPass* CGWP = &(getAnalysis<CallGraphWrapperPass>());
	for (auto recFunc : function_tasks)
	{
		if (!isRecursive(*recFunc, CGWP->getCallGraph()))
			continue;

		CallGraphNode *CGN;
		for (Module::iterator F = M.begin(); F != M.end(); ++F)
		{
			if (isRecursive(*F, CGWP->getCallGraph()) || F->empty())
				continue;

			CGN = (*CGWP)[F];
			for (unsigned i = 0; i < CGN->size(); i++)
			{
				auto caller = (*CGN)[i]->getFunction();
				if (caller == recFunc)
				{
					findTopLevelFunctionCall(*recFunc, *F);
				}
			}
		}
	}
}

//Traverse through all the function calls in caller until it finds callee
//then it adds it to the list of call insts.
void TaskMiner::findTopLevelFunctionCall(Function &callee, Function &caller)
{
	for (Function::iterator BB = caller.begin(); BB != caller.end(); ++BB)
		for (BasicBlock::iterator inst = BB->begin(); inst != BB->end(); ++inst)
		{
			CallInst* CI = dyn_cast<CallInst>(inst);
			if (!CI || CI->getCalledFunction() != &callee)
				continue;

			topLevelRecCalls.insert(CI);			
		}
}

//Check if function is recursive.
bool TaskMiner::isRecursive(Function &F, CallGraph &CG)
{
	auto callNode = CG[&F];
	for (unsigned i = 0; i < callNode->size(); i++)
	{
		if ((*callNode)[i]->getFunction() == &F)
			return true;
	}

	return false;
}

std::map<Function*, RegionTree*> TaskMiner::getAllRegionTrees(Module &M)
{
	std::map<Function*, RegionTree*> RTs;
	for (Module::iterator F = M.begin(); F != M.end(); ++F)
	{
		if (F->empty())
			continue;
		DepAnalysis* DP = &(getAnalysis<DepAnalysis>(*F));
		RTs[F] = std::move((DP->getRegionTree()));
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
		// F->dump();
		if (F->empty())
		{
			// errs() << "function with no body.\n";
			continue;
		}
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
	errs() << "\tMining function call tasks...\n";
	//For each FCALL edge, check if
	// A) it comes from an SCC
	// B) it goes to an SCC
	// C) src and dst are different functions
	bool indirectRecursion;
	std::set<CallInst*> callInstTasks;
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

			if (!CI)
				continue;
			
			if (function_tasks.find(CI->getParent()->getParent()) != function_tasks.end())
				continue;

			if (topLevelRecCalls.find(CI) != topLevelRecCalls.end())
				continue;

			bool hasLoop = srcRW->hasLoop;
			if ((srcRW->F != dstRW->F)
				&& (hasLoop)
				&& (!indirectRecursion)
				&& (!CI->getCalledFunction()->isIntrinsic())
				&& (!CI->getCalledFunction()->isDeclaration())
				&& (CI->getCalledFunction()->empty() == false)
				&& (CI->getCalledFunction()->getReturnType()->isVoidTy())
				/*&& (taskGraph->nodeReachesSCC(dstRW))*/)
			{
				Task* TASK = new FunctionCallTask(CI);
				tasks.push_back(TASK);
				NTASKS++;
				NFCALLTASKS++;
				callInstTasks.insert(CI);
			}
		}
	}

	for (auto ci : callInstTasks)
	{
		function_tasks.insert(ci->getCalledFunction());
		function_tasks.insert(ci->getParent()->getParent());
	}
}

//Updates the set of functions that have been covered by a kind of task. Any kind.
void TaskMiner::updateCoveredFunctions(std::set<Function*> &set)
{
	for (auto n : set)
		function_tasks.insert(n);
}

void TaskMiner::mineRecursiveTasks()
{
	errs() << "\tMining recursive call tasks...\n";
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


	std::map<Function*, std::list<CallInst*> > rec_calls;
	std::set<CallInst*> rec_calls_aux;

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
					if (calledF != F)
						continue;

					if (rec_calls_aux.find(CI) != rec_calls_aux.end())
						continue;

					rec_calls_aux.insert(CI);
					rec_calls[F].push_back(CI);
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

		//WHY DID I WRITE THIS? I don't remember this above /
	}
}

void TaskMiner::mineRegionTasks()
{
	errs() << "\tMining Region Tasks...\n";
	//1: go through every SCC >= 2
	//2: check the function in every SCC. only analyse those which
	//haven't been added as a functioncall task or recursive task
	//3: Find the largest region that covers that SCC
	//4: Build RegionTask with those BB's that are within the largest region
	std::set<Graph<RegionWrapper*, EdgeDepType>* > candidate_sccs;

	for (auto scc : SCCs)
	{
		if (scc->size() <= 1)
			continue;
		for (auto n : scc->getNodes())
		{
			if (std::find(function_tasks.begin(), 
				function_tasks.end(), n->getItem()->F) == function_tasks.end())
			{
				candidate_sccs.insert(scc);
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
		LoopInfoWrapperPass* LIWP = &(getAnalysis<LoopInfoWrapperPass>(*F));
		LoopInfo* LI = &(LIWP->getLoopInfo());

		//errs() << "\nAnalyizing function " << F->getName() << "\n";

		for (auto l : TMU.getLoopsInPreorder(LI))
		{
			// errs() << "\n\n";
			// l->dump();
			if (l->getLoopDepth() == 1)
			{
				RegionTask* TASK = new RegionTask();
				TASK->setHeaderBB(l->getHeader());
				for (auto BB : l->getBlocks())
				{
					TASK->addBasicBlock(BB);
				}
				tasks.push_back(TASK);
				NTASKS++;
				NREGIONTASKS++;
			}
		}

		// errs() << "\n\n";

		// RegionInfoPass* RIP = &(getAnalysis<RegionInfoPass>(*F));

		// RegionInfo* RI = &RIP->getRegionInfo();
		// Region* R = RI->getRegionFor(firstRW->entry);
		// for (auto node : scc->getNodes())
		// {
		// 	auto currentRW = node->getItem();
		// 	Region* currentRegion = RI->getRegionFor(currentRW->entry);
		// 	R = RI->getCommonRegion(R, currentRegion);
		// }

		// RegionTask* TASK = new RegionTask();

		// //Now with the region R in hands, let's build the regiontask!
		// for (Function::iterator BB = F->begin(); BB != F->end(); ++BB)
		// {
		// 	if (R->contains(BB) && RI->getRegionFor(BB) != R)
		// 	{
		// 		Region *R_ = RI->getRegionFor(BB);
		// 		TASK->addBasicBlock(BB);
		// 		TASK->addBasicBlock(R_->getExit());
		// 		TASK->setHeaderBB(R_->getEntry());
		// 	}
		// }

		// tasks.push_back(TASK);
		// NTASKS++;
		// NREGIONTASKS++;
	}
}



void TaskMiner::mineLoopTasks()
{

}

void TaskMiner::mineTasks(Module &M)
{
	mineRecursiveTasks();
	//Walk through every rec call and find the top level call.
	//Only add it if they are NOT inside a loop.
	determineTopLevelRecursiveCalls(M);	

	//Only mine the function calls that are NOT top level rec calls.
	mineFunctionCallTasks();

	mineRegionTasks();
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

void TaskMiner::resolvePrivateValues()
{
	for (auto task : tasks)
		task->resolvePrivateValues();
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
	{
		if (task->getKind() == Task::TaskKind::RECURSIVE_TASK)
			continue;
		task->computeCost();
	}
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
