#include "llvm/Support/CommandLine.h"
#include "DepAnalysis.h"
#define DEBUG_TYPE "loops-data"

using namespace llvm;

static cl::opt<bool, false> printToDot("printToDot",
  cl::desc("Print dot file containing the depgraph"), cl::NotHidden);

STATISTIC(RARDeps, "RAR Total num of input dependences");
STATISTIC(WARDeps, "WAR Total num of anti dependences");
STATISTIC(RAWDeps, "RAW Total num of flow/true dependences");
STATISTIC(WAWDeps, "RAR Total num of output dependences");
STATISTIC(SCADeps, "SCA Total num of scalar dependences");
STATISTIC(CTRDeps, "CTR Total num of control dependences");

int DepAnalysis::numberOfLoopData = 0;

void DepAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
	AU.setPreservesAll();
	AU.addRequired<PostDominatorTree>();
	AU.addRequired<DependenceAnalysis>();
	AU.addRequired<DominanceFrontier>();
	AU.addRequired<LoopInfoWrapperPass>();
	AU.addRequired<RegionInfoPass>();
}

bool DepAnalysis::runOnFunction(Function &F) {

	LoopInfoWrapperPass *LIWP = &(getAnalysis<LoopInfoWrapperPass>());
	LI = &(LIWP->getLoopInfo());

	RegionInfoPass *RIP = &(getAnalysis<RegionInfoPass>());
	RI = &(RIP->getRegionInfo());

	//Step1: Create PDG
	createProgramDependenceGraph(F);
	
	//Step2: Get info about the loops
	getLoopsInfo(F);
	
	//Step3: Find windmills
	SCCs = G->findStrongConnectedComponents();
	findWindmills();
	
	//Step4: For each Windmill, find helices
	findHelices();

	findMinimumRegionForEachHelix();

	// DEBUG_WITH_TYPE("loop-data", printWindmillsRegions());
	DEBUG_WITH_TYPE("loop-data", print());
	// DEBUG_WITH_TYPE("loop-data", dumpWindmillsToDot(F));
	if (printToDot)	
		G->dumpToDot(F);

	return true;
}

void DepAnalysis::findMinimumRegionForEachHelix()
{
	std::set<Instruction*> insts;

	for (auto &l : loops)
	{
		for (auto &node : l.second->W->H->subgraph)
		{
			insts.insert(node->instr);
		}

		Region *R = getMinimumCoverRegion(insts);
		l.second->R = R;

		insts.clear();
	}
}

void DepAnalysis::createProgramDependenceGraph(Function &F) 
{
	auto &DI = getAnalysis<DependenceAnalysis>();
	auto &PDT = getAnalysis<PostDominatorTree>();

	// This is going to represent the graph
	G = new ProgramDependenceGraph(F.getName());

	// Add all nodes to the graph
	for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
		for (BasicBlock::iterator i = BB->begin(), e = BB->end(); i != e; ++i)
			G->addNode(&*i);

	// Collect memory dependence edges
	for (inst_iterator SrcI = inst_begin(F), SrcE = inst_end(F); SrcI != SrcE; ++SrcI) {

		if (isa<StoreInst>(*SrcI) || isa<LoadInst>(*SrcI)) {

			for (inst_iterator DstI = SrcI, DstE = inst_end(F); DstI != DstE; ++DstI) {
				
				if (isa<StoreInst>(*DstI) || isa<LoadInst>(*DstI)) {

					DEBUG_WITH_TYPE("dep-analysis", errs() << "\nChecking [" << *SrcI << "] and [" << *DstI << "]\n");

					if (auto D = DI.depends(&*SrcI, &*DstI, true)) {
						if (D->isInput())
						{
							G->addEdge(&*SrcI, &*DstI, DependenceType::RAR);
							RARDeps++;
						}
						else if (D->isOutput())
						{
							G->addEdge(&*SrcI, &*DstI, DependenceType::WAW);
							WAWDeps++;
						}
						else if (D->isFlow())
						{
							G->addEdge(&*SrcI, &*DstI, DependenceType::RAW);
							RAWDeps++;
						}
						else if (D->isAnti())
						{
							G->addEdge(&*DstI, &*SrcI, DependenceType::RAWLC);
							WARDeps++;
						}
						else
							DEBUG_WITH_TYPE("dep-analysis", errs() << "Error decoding dependence type.\n");

						DEBUG_WITH_TYPE("dep-analysis", errs() << "\tDependent.\n");
					}
					else
						DEBUG_WITH_TYPE("dep-analysis", errs() << "\tIndependent.\n");
				}
			}
		}
	}

	// Collect data dependence edges
	for (inst_iterator SrcI = inst_begin(F), SrcE = inst_end(F); SrcI != SrcE; ++SrcI) {
		for (User *U : SrcI->users()) {
			if (Instruction *Inst = dyn_cast<Instruction>(U)) {
				G->addEdge(&*SrcI, Inst, DependenceType::SCA);
				SCADeps++;
			}
		}
	}

	// Collect control dependence edges
	ControlDependenceGraphBase cdgBuilder;
	cdgBuilder.graphForFunction(F, PDT);
	G->addEntryNode((Instruction*)1000000);
	G->addExitNode((Instruction*)2000000);

	for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
		BasicBlock *A = &*BB;
		auto term = dyn_cast<Instruction>(A->getTerminator());

		for (Function::iterator BB2 = F.begin(), E2 = F.end(); BB2 != E2; ++BB2) {
			BasicBlock *B = &*BB2;

			if (cdgBuilder.controls(A, B)) {
				for (BasicBlock::iterator i = B->begin(), e = B->end(); i != e; ++i) {
					G->addEdge(term, &*i, DependenceType::CTR);
					CTRDeps++;
				}
			}
		}

		if (!cdgBuilder.isDominated(A)) {
			for (BasicBlock::iterator i = A->begin(), e = A->end(); i != e; ++i) {
				G->connectToEntry(&*i);
			}
		}

		if (cdgBuilder.getNode(A)->getNumChildren() == 0) {
			for (BasicBlock::iterator i = A->begin(), e = A->end(); i != e; ++i) {
				G->connectToExit(&*i);
			}
		}
	}
}

ProgramDependenceGraph* DepAnalysis::getDepGraph() { return G; }

void DepAnalysis::getLoopsInfo(Function &F)
{
	//iterate over basic blocks and find all loops
	//Find all Loops
	for (Function::iterator BB = F.begin(); BB != F.end(); ++BB)
	{
		//If BB is in loop, then we can try to get the induction var from this loop
		Loop* loop = LI->getLoopFor(BB);
		if (loop)
		{
			if (loops.find(loop) == loops.end()) //If the loop hasn't been added to the loopdata yet
			{
				LoopData* LD = new LoopData();
				LD->id = numberOfLoopData;
				numberOfLoopData++;
				loops[loop] = LD;
				Instruction* inst = loop->getCanonicalInductionVariable();
				loops[loop]->indVar = inst;
			}
		}
	}
}

void DepAnalysis::findWindmills()
{
	//find windmills -> SCC's with loop's ind var
	std::vector<std::set<std::set<GraphNode*>>::iterator > toBeRemoved;
	for (auto scc = SCCs.begin(); scc != SCCs.end(); scc++)
	{
		//if SCC contains loop ind var and it's a windmill centre.
		if ((*scc).size() > 2)
		{
			for (auto &node : *scc)
			{
				for (auto &l : loops)
				{
					if (l.second->indVar != nullptr 
							&& l.second->indVar == node->instr)
					{
						Windmill* W = new Windmill();
						toBeRemoved.push_back(scc);
						W->nodes = *scc;
						l.second->W = W;
					}
				}
			}			
		}
	}

	for (const auto tbr : toBeRemoved)
	{
		SCCs.erase(tbr);
	}
	toBeRemoved.clear();

	//if loop's ind var is null, let's try to find it geographically.
	//we look for scc's of size 4, with phi/arithmetic/compare/branch inst
	//then we take the basic blocks of this four insts, check if the basic
	//blocks are inside loop. If they are, then check if phi node is ind var
	//from a different loop already analysed. If after that more than one candidate
	//exists, it's okay, just pick a random one, we're interested in the 
	//SCC, not in the ind var.
	std::set<std::set<GraphNode*> > candidateSCCs;
	bool foundPattern = true;
	for (auto &scc : SCCs)
	{
		if (scc.size() == 4)
		{
			for (auto &node : scc)
			{
				if (!((isa<PHINode>(node->instr))
					|| (isa<BranchInst>(node->instr))
					|| (isa<BinaryOperator>(node->instr))
					|| (isa<CmpInst>(node->instr))))
				{
					foundPattern = false;
				}
			}
			if (foundPattern)
			{
				candidateSCCs.insert(scc);
			}
		}
	}

	//Look for loops with these candidate SCC's
	std::map<Loop*, std::set<GraphNode*> > loopIndVarSCCs;
	bool found;
	for (const auto &scc : candidateSCCs)
	{
		for (const auto &l : loops)
		{
			found = false;
			if (!l.second->indVar)
			{
				for (const auto &node : scc)
				{
					auto bbs = l.first->getBlocks();
					auto bb = node->instr->getParent();
					for (BasicBlock* b : bbs)
					{
						if (b == bb)
						{
							found = true;
							break;
						}
					}
				}
				//TODO: WHAT ABOUT WHEN I FIND TWO OR MORE SCCS THAT FIT THAT PATTERN
				//FOR THE SAME LOOP? 
				if (found)
				{
					loopIndVarSCCs[l.first] = scc;
				}				
			}
		}
	}
	
	std::vector<std::map<Loop*, std::set<GraphNode*> >::iterator > toBeRemoved2;
	for (std::map<Loop*, std::set<GraphNode*> >::iterator l = loopIndVarSCCs.begin();
		l != loopIndVarSCCs.end(); l++) 
	{
		loops[l->first]->W = new Windmill();
		loops[l->first]->W->nodes = l->second;
		toBeRemoved2.push_back(l);
	}

	//Also remove these from the SCC's list
	for (const auto tbr : toBeRemoved2)
	{
		SCCs.erase(tbr->second);
	}
}

void DepAnalysis::findHelices()
{
	std::set<GraphNode*> nodeSet;
	for (auto &l : loops)
	{
		Windmill* W_ = l.second->W;
		auto scc = W_->nodes;
		if (scc.empty())
		{
			errs() << "Trying to find Helices but Windmill has got no nodes.";
			return;
		}
		G->getSubgraphOnSCC(scc, nodeSet);
		if (!nodeSet.empty())
		{
			Helix* H = new Helix();
			H->subgraph = nodeSet;
			W_->H = H;				
		}
		nodeSet.clear();
	}
	//check if helices contain any SCC's that ARE NOT the indvars scc's

}

Region *DepAnalysis::getMinimumCoverRegion(std::set<Instruction*> insts)
{
	Region *R;
	std::set<Region*> regions;
	std::set<BasicBlock*> bbs;

	for (auto &i : insts)
	{
		bbs.insert(i->getParent());
	}

	for (auto &bb : bbs)
	{
		regions.insert(RI->getRegionFor(bb));
	}

	Region* r1 = *(regions.begin());
	for (auto &r2 : regions)
	{
		r1 = RI->getCommonRegion(r1, r2);
	}

	return r1;
}


raw_ostream& DepAnalysis::print(raw_ostream& os) const
{
	os << "\nPrinting LoopData";
	for (const auto &l : loops)
	{
		os <<"\n=============================";
		os << "\nLOOPDATA\n " << *l.first;
		l.second->print(os);
		if (l.second->R)
		{
		os << "\nREGION:\n";
		l.second->R->print(errs());			
		}
		os <<"\n=============================\n";
	}

	return os;
}

void DepAnalysis::dumpWindmillsToDot(Function &F)
{
	int i = 0;
	for (auto &l : loops)
	{
		dumpWindmillsToDot(F, *(l.second->W), i);
		i++;
	}
} 

void DepAnalysis::dumpWindmillsToDot(Function &F, Windmill &W, int windmillId) 
{
	// Write the graph to a DOT file
	std::string functionName = F.getName();
	std::string graphName = functionName + "-windmill-" + std::to_string(windmillId) + ".dot";
	std::ofstream dotStream;
	dotStream.open(graphName);

	if (!dotStream.is_open()) {
		errs() << "Problem opening DOT file: " << graphName << "\n";
	}	
	else {
		dotStream << "digraph g {\n";

		// Create all nodes in DOT format
		for (auto node : G->instrToNode) {
			if (node.second.second == G->entry) 
				dotStream << "\t\"" << G->instrToNode[node.second.second->instr].first << "\" [label=entry];\n";
			else if (node.second.second == G->exit)
				dotStream << "\t\"" << G->instrToNode[node.second.second->instr].first << "\" [label=exit];\n";
			else
				dotStream << "\t\"" << G->instrToNode[node.second.second->instr].first << "\" [label=\"" << G->instrToNode[node.second.second->instr].first << ": " << (node.second.second->instr->getOpcodeName()) << "\"];\n";
		}

		dotStream << "\n\n";

		// Now print all outgoing edges and their labels
		for (auto src : G->outEdges) {
			for (auto dst : src.second) {
				if (dst->type == CTR)
					dotStream << "\t\"" << G->instrToNode[src.first->instr].first << "\" -> \"" << G->instrToNode[dst->dst->instr].first << "\" [style=dotted];\n";
				else
					dotStream << "\t\"" << G->instrToNode[src.first->instr].first << "\" -> \"" << G->instrToNode[dst->dst->instr].first << "\" [label=\"" << dst->edgeLabel() << "\"];\n";
			}

			dotStream << "\n";
		}

		//print SCCS
	  std::set<GraphNode *> q;
	  GraphNode *node;
		int i=0;
		// for (std::set<std::set<GraphNode *>>::iterator it = SCCs.begin();
  //      it != SCCs.end(); ++it)
	 //  {
	 //  	if ((*it).size() <= 1)
	 //  		continue;
	 //    dotStream << "\nsubgraph cluster_" << i << " {\n"
	 //              << " color=red4;"
	 //              << " label=SCC" << i + 1 << ";"
	 //              << " fillcolor=paleturquoise1;"
	 //              << " style=filled;\n";
	 //    i++;
	 //    q = (*it);
  //     std::set<GraphNode *>::iterator s = q.begin();
  //     do
  //     {
  //       node = *s;
  //       dotStream << G->instrToNode[node->instr].first;
  //       s++;
  //       if (s != q.end())
  //       {
  //         dotStream << ",";
  //       }
  //     } while (s != q.end());
  //     dotStream << ";";

	 //    dotStream << "\n} ";
	 //  }

	  //printWindmills
	  int helixCount = 1;
	  int windmillCount = 1;
    std::set<GraphNode*> scc_ = W.nodes;
    dotStream << "\nsubgraph cluster_" << i << " {\n"
              << " color=red4;"
              << " label=WINDMILL" << windmillCount << ";"
              << " fillcolor=orange;"
              << " style=filled;\n";
    i++;
    q = W.nodes;
    if (q.size() > 1)
    {
      std::set<GraphNode *>::iterator s = q.begin();
      do
      {
        node = *s;
        dotStream << G->instrToNode[node->instr].first;
        s++;
        if (s != q.end())
        {
          dotStream << ",";
        }
      } while (s != q.end());
      dotStream << ";";
    }

    dotStream << "\n} ";
    windmillCount++;

    //printHelices
  	auto h = W.H;
		dotStream << "\nsubgraph cluster_" << i << " {\n"
              << " color=red4;"
              << " label=helix" << helixCount << ";"
              << " fillcolor=wheat;"
              << " style=filled;\n";
    i++;
    q = h->subgraph;
    if (q.size() > 1)
    {
	    auto s = q.begin();
	    bool empty = true;
	    do
	    {
	      node = *s;
	      if (std::find(std::begin(scc_), std::end(scc_), node) == std::end(scc_))
	      {
	      	empty = false;
	        dotStream << G->instrToNode[node->instr].first;
	      }
	      s++;
	      if (!empty)
	      {
	        if (s != q.end())
		      {
	  	      dotStream << ",";
	  	      empty = true;
	    	  }
	      }
	    } while (s != q.end());
	    if (!empty)
	      dotStream << ";";
	    dotStream << "\n} ";
	    helixCount++;    	
    }
	  

		dotStream << "\n}}";

		dotStream.close();
	
	}
}

raw_ostream& LoopData::print(raw_ostream& os) const
{
	os <<"# " << id << "\n";
	os << "INDVAR: ";
	if (indVar)
		indVar->print(os);
	else
		os << "nullptr ";
	os << "\nRegular ? " << (regular ? "yes" : "no");
	if (W)
		W->print(os);

	return os;
}

raw_ostream& Windmill::print(raw_ostream& os) const
{
	os << "\nWINDMILLDATA";
	os << "\nCentre Nodes: \n";
	for (const auto &n : nodes)
	{
		if (const auto i = n->instr)
			i->print(os);
		os << "\n";
	}
	os << "\nHELIX:\n";
	H->print(os);

	return os;
}

raw_ostream& Helix::print(raw_ostream& os) const
{
	for (auto n : subgraph)
	{
		if (n->instr != nullptr)
			n->instr->print(os);
		os << "\n";
	}

	return os;
}

char DepAnalysis::ID = 0;
static RegisterPass<DepAnalysis> X("depanalysis", "Run the DepAnalysis algorithm. Generates a dependence graph", false, false);
