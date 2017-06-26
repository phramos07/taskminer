//===------------------------------- TaskMiner.h --------------------------===//
//
// Author: Pedro Ramos (pedroramos@dcc.ufmg.br)
//
//===----------------------------------------------------------------------===//
//
//                           The LLVM Compiler Infrastructure
//
//
//===----------------------------------------------------------------------===//
#ifndef TASKMINER_H
#define TASKMINER_H

//LLVM IMPORTS
#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

//LOCAL IMPORTS
#include "DepAnalysis.h"
#include "Task.h"

//STL IMPORTS
#include <list>
#include <set>
#include <iostream>
#include <string>

namespace llvm
{
	class TaskMiner : public FunctionPass
	{
	public:
		static char ID;
		TaskMiner() : FunctionPass(ID) {}
		~TaskMiner();
		void getAnalysisUsage(AnalysisUsage &AU) const override;
		bool runOnFunction(Function &func) override;
		bool doFinalization(Module &M) override;
		std::list<Task*> getTasks();

		struct LoopData
		{
			Instruction* indVar;
			std::list<BasicBlock*> innerBBs;
			bool regular;

			//Debugginf purposes only
			void print();
		};

	private:
		std::list<Task*> tasks;
		void resolveInsAndOutsSets();
		void getStats();
		void getLoopsInfo(Function &F);
		void mineFunctionCallTasks(Function &F);
		void insertNewFunctionCallTask(Task &T);

		std::map<Loop*, TaskMiner::LoopData> loops;
		DepAnalysis* DA = 0;
		LoopInfo* LI = 0;
		LoopInfoWrapperPass* LIWP = 0;

		//Debugging purposes only
		void printLoops();
	};

}

#endif
