#include "DepAnalysis.h"

using namespace llvm;		

void DepAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
	AU.setPreservesAll();
	AU.addRequired<PostDominatorTree>();
	AU.addRequired<DependenceAnalysis>();
	AU.addRequired<DominanceFrontier>();
}

bool DepAnalysis::runOnFunction(Function &F) {
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

					errs() << "\nChecking [" << *SrcI << "] and [" << *DstI << "]\n";

					if (auto D = DI.depends(&*SrcI, &*DstI, true)) {
						if (D->isInput())
							G->addEdge(&*SrcI, &*DstI, DependenceType::RAR);
						else if (D->isOutput())
							G->addEdge(&*SrcI, &*DstI, DependenceType::WAW);
						else if (D->isFlow())
							G->addEdge(&*SrcI, &*DstI, DependenceType::RAW);
						else if (D->isAnti()) {
							G->addEdge(&*DstI, &*SrcI, DependenceType::RAWLC);
						}
						else
							errs() << "Error decoding dependence type.\n";

						errs() << "\tDependent.\n";
					}
					else
						errs() << "\tIndependent.\n";
				}
			}
		}
	}

	// Collect data dependence edges
	for (inst_iterator SrcI = inst_begin(F), SrcE = inst_end(F); SrcI != SrcE; ++SrcI) {
		for (User *U : SrcI->users()) {
			if (Instruction *Inst = dyn_cast<Instruction>(U)) {
				G->addEdge(&*SrcI, Inst, DependenceType::SCA);
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

	// G->dumpToDot(F);

	return true;
}

ProgramDependenceGraph* DepAnalysis::getDepGraph() { return G; }

char DepAnalysis::ID = 0;
static RegisterPass<DepAnalysis> X("depanalysis", "Run the DepAnalysis algorithm. Generates a dependence graph", false, false);
