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
		~TaskMiner() {};
		void getAnalysisUsage(AnalysisUsage &AU) const override;
		bool runOnFunction(Function &func) override;
		std::list<Task*>* getTasks();
		Task* getTaskFromParentLoop(Loop* L);

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
		DepAnalysis *DA;
	};

	class Task
	{
	public:
		enum TaskKind
		{
			FCALL_TASK,
			CFRAGMENT_TASK,
			NLOOP_TASK
		};

		Task(TaskKind k, Loop* p) : kind(k), parent(p) {};
		virtual ~Task() {};
		TaskKind getKind() const { return kind; };
		Loop* getParent();
		virtual bool resolveInsAndOutsSets() { return false; };
		std::set<Value*> getLiveIN() const;
		std::set<Value*> getLiveOUT() const;
		std::set<Value*> getLiveINOUT() const;

		//Only for debugging purposes. STDOUT/errs()
		virtual void print();
		void printLiveSets();

	protected:
		Loop* parent;
		std::set<Value*> liveIN;
		std::set<Value*> liveOUT;
		std::set<Value*> liveINOUT;
		AccessType getTypeFromInst(Instruction* I);
		std::string accessTypeToStr(AccessType T);	

	private:
		const TaskKind kind;
	};

	class FunctionCallTask : public Task
	{
	public:
		FunctionCallTask(Loop* parent, CallInst* CI) : Task(FCALL_TASK, parent), functionCall(CI) {};
		~FunctionCallTask() {};
		CallInst* getFunctionCall() { return functionCall; };
		bool resolveInsAndOutsSets() override;
		void print() override;
		
		static bool classof(const Task* T) { return T->getKind() == FCALL_TASK; };
	
	private:
		CallInst* functionCall;
	};

}

#endif