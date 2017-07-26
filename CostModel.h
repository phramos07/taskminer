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

		float getCost()
		{
			return nInsts/minInstCost;
		}

		float getThreshold()
		{
			return THRESHOLD;
		}

		bool aboveThreshold()
		{
			return getCost() >= getThreshold();
		}
	};
}

