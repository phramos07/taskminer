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

//LOCAL IMPORTS
#include "DepAnalysis.h"
#include "LiveSets.h"

//STL IMPORTS
#include <list>
#include <set>
#include <iostream>
#include <string>

namespace llvm{

	class Task;
	class FunctionCallTask;
	class NestedLoopTask;
	class CodeFragmentTask;

	class TaskMiner : public FunctionPass
	{
	public:

		static char ID;
		TaskMiner() : FunctionPass(ID) {}
		~TaskMiner() {};
		void getAnalysisUsage(AnalysisUsage &AU) const override;
		bool runOnFunction(Function &func) override;
		std::list<Task*>* getTasks();
		Task* getTaskFromParentLoop(Loop* L);

	private:
		std::list<Task*> tasks;
		
	};

	class Task
	{
	public:
		Task(Loop* parent) : parent(parent) {};
		~Task();
		std::set<Instruction*>* getInstructions() const;
		Loop* getParent();
		virtual bool resolveInsAndOutsSets();

	protected:
		Loop* parent;
		std::set<Instruction*> instructions;
		std::set<Value*> liveIN;
		std::set<Value*> liveOUT;

		
	};

	class FunctionCallTask : public Task
	{
	public:
		FunctionCallTask(Loop* parent) : Task(parent) {};
		~FunctionCallTask();
		CallInst* getFunctionCall();
		bool resolveInsAndOutsSets() override;
		
	private:
		CallInst* functionCall;
		void matchFormalParametersWithArguments();



	};

}

#endif