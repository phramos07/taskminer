#ifndef TASK_H
#define TASK_H

#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/RegionInfo.h"

//Local imports
#include "CostModel.h"

#include <set>
#include <stack>

namespace llvm
{
	class Task;
	class FunctionCallTask;
	class RegionTask;
	class RecursiveTask;

	enum AccessType {UNKNOWN=0, READ=1, WRITE=2, READWRITE=3};

	inline AccessType operator|(AccessType a, AccessType b)
	{
		return static_cast<AccessType>(static_cast<int>(a) | static_cast<int>(b)); 
	}

	class Task
	{
	public:
		enum TaskKind
		{
			FCALL_TASK,
			REGION_TASK,
			RECURSIVE_TASK
		};

		Task(TaskKind k) 
			: kind(k)
			{}
		virtual ~Task() {};
		
		//Getters
		TaskKind getKind() const { return kind; }
		std::set<Value*> getLiveIN() const;
		std::set<Value*> getLiveOUT() const;
		std::set<Value*> getLiveINOUT() const;
		std::set<BasicBlock*> getbbs() const;
		CostModel getCost() const { return CM; }

		//Methods
		virtual bool resolveInsAndOutsSets() { return false; }
		virtual CostModel computeCost() { return CM; }
		void addBasicBlock(BasicBlock* bb) { bbs.insert(bb); }

		//Printing to output stream methods
		virtual raw_ostream& print(raw_ostream& os) const;
		raw_ostream& printLiveSets(raw_ostream& os) const;

	private:
		const TaskKind kind;

	protected:
		//Cost model
		CostModel CM;

		//Content:
		std::set<BasicBlock*> bbs;
		std::set<Value*> liveIN;
		std::set<Value*> liveOUT;
		std::set<Value*> liveINOUT;

		//Private methods
		AccessType getTypeFromInst(Instruction* I);
		std::string accessTypeToStr(AccessType T);
		bool isPointerValue(Value *V);

	};

	class FunctionCallTask : public Task
	{
	public:
		FunctionCallTask(CallInst* CI);
		~FunctionCallTask() {}
		CallInst* getFunctionCall() const;
		bool resolveInsAndOutsSets() override;
		CostModel computeCost() override;
		raw_ostream& print(raw_ostream& os) const override;
		static bool classof(const Task* T) { return T->getKind() == FCALL_TASK; }

	private:
		CallInst* functionCall;
	};

	class RegionTask : public Task
	{
	public:
		RegionTask() 
			: Task(REGION_TASK)
			{}
		~RegionTask() {}
		bool resolveInsAndOutsSets() override;
		CostModel computeCost() override;
		raw_ostream& print(raw_ostream& os) const override;
		static bool classof(const Task* T) { return T->getKind() == REGION_TASK; }
		
	};

	class RecursiveTask : public Task
	{
	public:
		RecursiveTask(CallInst *CI, bool isInsideLoop);
		~RecursiveTask() {}
		CallInst* getRecursiveCall() const;
		bool resolveInsAndOutsSets() override;
		CostModel computeCost() override;
		raw_ostream& print(raw_ostream& os) const override;
		static bool classof(const Task* T) { return T->getKind() == RECURSIVE_TASK; }		
		RecursiveTask* getPrev() const { return prev; }
		RecursiveTask* getNext() const { return next; }
		void setPrev(RecursiveTask* prev) { this->prev = prev; }
		void setNext(RecursiveTask* next) { this->next = next; }
		bool hasSyncBarrier() { if (prev && !next) return true; else return false;}
		bool insideLoop() { return isInsideLoop; }

	private:
		RecursiveTask* prev=nullptr;
		RecursiveTask* next=nullptr;
		CallInst *recursiveCall;
		bool isInsideLoop=false;
	};
}

#endif