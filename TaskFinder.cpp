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
using namespace llvm;

#include <set>
#include <map>
#include <string>
#include <fstream>
#include <iomanip>

INITIALIZE_PASS_BEGIN(DependenceAnalysisWrapperPass, "TaskFinder", "Run the TaskFinder algorithm", true, true)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DependenceAnalysisWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominanceFrontierWrapperPass);
INITIALIZE_PASS_DEPENDENCY(PostDominatorTreeWrapperPass);
INITIALIZE_PASS_END(DependenceAnalysisWrapperPass, "TaskFinder", "Run the TaskFinder algorithm", true, true)

namespace {
	// Only edge type are necessary for now. We don't keep track of distances.
	enum DependenceType {RAR, WAW, RAW, WAR, CTR, SCA, RAWLC};

	inline std::string getDependenceName(DependenceType V) {
		switch (V) {
			case RAR: return "RAR";
			case RAWLC: return "RAW*";
			case WAW: return "WAW";
			case RAW: return "RAW";
			case WAR: return "WAR";
			case CTR: return "CTR";
			case SCA: return "SCA";
			default: return std::to_string(V);
		}
	}

	inline llvm::raw_ostream & operator<<(llvm::raw_ostream & Str, DependenceType V) {
		return Str << getDependenceName(V);
	}



	struct TaskFinder : public FunctionPass {
		// Pass identification placeholder - required by LLVM
		static char ID;
		TaskFinder() : FunctionPass(ID) {}

		void getAnalysisUsage(AnalysisUsage &AU) const {
			AU.setPreservesAll();
			AU.addRequired<PostDominatorTreeWrapperPass>();
			AU.addRequiredTransitive<DependenceAnalysisWrapperPass>();
		}

		bool runOnFunction(Function &F) override {
			auto &DI = getAnalysis<DependenceAnalysisWrapperPass>().getDI();
			auto &PDT = getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree();

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

char TaskFinder::ID = 0;
static RegisterPass<TaskFinder> X("TaskFinder", "Run the TaskFinder algorithm");
