#ifndef _COST_MODEL_H_
#define _COST_MODEL_H_

//LLVM IMPORTS
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"

//LOCAL IMPORTS
#include "RegionTree.h"

namespace llvm
{
	class CostModel
	{
	private:
		uint32_t nInsts;
		uint32_t nCores;
		uint32_t nThreads;
		uint32_t nInDeps;
		uint32_t nOutDeps;
		static const uint32_t minInstCost = 500;
		static const float constexpr THRESHOLD = 2.0;

	public:
		CostModel() {};
		~CostModel() {};

		void setData(uint32_t nInsts, uint32_t nInDeps, uint32_t nOutDeps)
		{
			this->nInsts = nInsts;
			this->nInDeps = nInDeps;
			this->nOutDeps = nOutDeps;
		}

		float getCost() const
		{
			return (float)nInsts/(float)minInstCost;
		}

		float getThreshold()
		{
			return THRESHOLD;
		}

		bool aboveThreshold()
		{
			return getCost() >= getThreshold();
		}

		uint32_t getNInsts() { return nInsts; }
		uint32_t getNInDeps() { return nInDeps; }
		uint32_t getNOutDeps() { return nOutDeps; }

		raw_ostream& print(raw_ostream& os) const
		{
			os << "\nTASK COST:\n";
			os << nInsts << " instructions\n";
			os << nInDeps << " input dependencies\n";
			os << nOutDeps << " output dependencies\n";
			os << getCost() << " cost: (inst_count/min_inst_count)\n\n";
		}

	};
}

#endif // _COST_MODEL_H_