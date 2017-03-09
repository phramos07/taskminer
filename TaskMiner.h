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

//STL IMPORTS
#include <list>
#include <set>
#include <iostream>
#include <string>

namespace llvm
{
	class Task;
	class FunctionCallTask;
	// class NestedLoopTask;
	// class CodeFragmentTask;
	// class RecursiveTask;

	enum AccessType {UNKNOWN=0, READ=1, WRITE=2, READWRITE=3};

	inline AccessType operator|(AccessType a, AccessType b)
	{
		return static_cast<AccessType>(static_cast<int>(a) | static_cast<int>(b)); 
	}

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

	class Task
	{
	public:
		enum TaskKind
		{
			FCALL_TASK,
			CFRAGMENT_TASK,
			NLOOP_TASK,
			RECURSIVE_TASK
		};

		Task(TaskKind k, Loop* p) : kind(k), parent(p) {};
		virtual ~Task() {};
		
		//Getters
		TaskKind getKind() const { return kind; };
		Loop* getParent();
		std::set<Value*> getLiveIN() const;
		std::set<Value*> getLiveOUT() const;
		std::set<Value*> getLiveINOUT() const;

		//Methods
		virtual bool resolveInsAndOutsSets() { return false; };

		//Printing to output stream methods
		virtual raw_ostream& print(raw_ostream& os) const;
		raw_ostream& printLiveSets(raw_ostream& os) const;
	
	private:
		const TaskKind kind;

	protected:
		Loop* parent;
		std::set<Value*> liveIN;
		std::set<Value*> liveOUT;
		std::set<Value*> liveINOUT;
		AccessType getTypeFromInst(Instruction* I);
		std::string accessTypeToStr(AccessType T);	
	};

	class FunctionCallTask : public Task
	{
	public:
		FunctionCallTask(Loop* parent, CallInst* CI) : Task(FCALL_TASK, parent), functionCall(CI) {};
		~FunctionCallTask() {};
		CallInst* getFunctionCall() const;
		bool resolveInsAndOutsSets() override;
		raw_ostream& print(raw_ostream& os) const override;

		static bool classof(const Task* T) { return T->getKind() == FCALL_TASK; };
	
	private:
		CallInst* functionCall;
	};



}


#endif
