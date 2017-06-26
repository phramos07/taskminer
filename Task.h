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

#include <set>

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
		Region* region;
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