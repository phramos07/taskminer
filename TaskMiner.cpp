#include "TaskMiner.h"

namespace {
	struct TaskMiner : public FunctionPass {
		// Pass identification placeholder - required by LLVM
		static char ID;
		TaskMiner() : FunctionPass(ID) {}

		void getAnalysisUsage(AnalysisUsage &AU) const {
			AU.setPreservesAll();

			AU.addRequired<PostDominatorTree>();
			AU.addRequired<DependenceAnalysis>();
			AU.addRequired<DominanceFrontier>();
		}

		bool runOnFunction(Function &F) override {
			auto &DI = getAnalysis<DependenceAnalysis>();
			auto &PDT = getAnalysis<PostDominatorTree>();

			// This is going to represent the graph
			ProgramDependenceGraph g(F.getName());

			// Add all nodes to the graph
			for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
				for (BasicBlock::iterator i = BB->begin(), e = BB->end(); i != e; ++i)
					g.addNode(&*i);

			// Collect memory dependence edges
			for (inst_iterator SrcI = inst_begin(F), SrcE = inst_end(F); SrcI != SrcE; ++SrcI) {

				if (isa<StoreInst>(*SrcI) || isa<LoadInst>(*SrcI)) {

					for (inst_iterator DstI = SrcI, DstE = inst_end(F); DstI != DstE; ++DstI) {
						
						if (isa<StoreInst>(*DstI) || isa<LoadInst>(*DstI)) {

							errs() << "\nChecking [" << *SrcI << "] and [" << *DstI << "]\n";

							if (auto D = DI.depends(&*SrcI, &*DstI, true)) {
								if (D->isInput())
									g.addEdge(&*SrcI, &*DstI, DependenceType::RAR);
								else if (D->isOutput())
									g.addEdge(&*SrcI, &*DstI, DependenceType::WAW);
								else if (D->isFlow())
									g.addEdge(&*SrcI, &*DstI, DependenceType::RAW);
								else if (D->isAnti()) {
									g.addEdge(&*DstI, &*SrcI, DependenceType::RAWLC);
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
						g.addEdge(&*SrcI, Inst, DependenceType::SCA);
					}
				}
			}


			// Collect control dependence edges
			ControlDependenceGraphBase cdgBuilder;

			cdgBuilder.graphForFunction(F, PDT);

			g.addEntryNode((Instruction*)1000000);
			g.addExitNode((Instruction*)2000000);

			for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
				BasicBlock *A = &*BB;
				auto term = dyn_cast<Instruction>(A->getTerminator());

				for (Function::iterator BB2 = F.begin(), E2 = F.end(); BB2 != E2; ++BB2) {
					BasicBlock *B = &*BB2;

					if (cdgBuilder.controls(A, B)) {
						for (BasicBlock::iterator i = B->begin(), e = B->end(); i != e; ++i) {
							g.addEdge(term, &*i, DependenceType::CTR);
						}
					}
				}

				if (!cdgBuilder.isDominated(A)) {
					for (BasicBlock::iterator i = A->begin(), e = A->end(); i != e; ++i) {
						g.connectToEntry(&*i);
					}
				}

				if (cdgBuilder.getNode(A)->getNumChildren() == 0) {
					for (BasicBlock::iterator i = A->begin(), e = A->end(); i != e; ++i) {
						g.connectToExit(&*i);
					}
				}
			}

			g.dumpToDot(F);

			return true;
		}
	};
}

char TaskMiner::ID = 0;
static RegisterPass<TaskMiner> X("TaskMiner", "Run the TaskMiner algorithm", false, false);
