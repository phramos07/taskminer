#include "ValueAccessTypes.h"

using namespace llvm;

bool ValueAccessTypes::runOnFunction(Function &F) {
	auto RA = &getAnalysis<lge::PtrRangeAnalysis>();
  auto RI = &getAnalysis<RegionInfoPass>().getRegionInfo();

	RA->analyzeRegionPointers(RI->getTopLevelRegion());
	errs()<< "\n\nPrinting Access Type\n\n";

	for (auto BB = F.begin(), BE = F.end(); BB != BE; BB++)
	{
		for (auto I = BB->begin(), IE = BB->end(); I != IE; I++)
		{
			if(isa<LoadInst>(I) || isa<StoreInst>(I) || isa<GetElementPtrInst>(I)) 
			{
			  Value *BasePtrV = lge::getPointerOperand(I);
			  while (isa<LoadInst>(BasePtrV) || isa<GetElementPtrInst>(BasePtrV)) 
			  {
			    if (LoadInst *LD = dyn_cast<LoadInst>(BasePtrV))
			      BasePtrV = LD->getPointerOperand();
			    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(BasePtrV))
			      BasePtrV = GEP->getPointerOperand();
			  }
			  I->dump();
			  errs() 	<< "\n" 
			  				<< (char)(RA->getPointerAcessType(RI->getTopLevelRegion(), BasePtrV) + '0') 
			  				<< "\n";
			}
		}		
	}

	return false;
}

void ValueAccessTypes::getAnalysisUsage(AnalysisUsage &AU) const {
	AU.addRequired<lge::PtrRangeAnalysis>();
  AU.addRequiredTransitive<RegionInfoPass>();
}

std::map<Value*, ValueAccessTypes::AccessType> 
ValueAccessTypes::getValuesAcessTypes() {
	return accessTypes;
}

char ValueAccessTypes::ID = 0;
static RegisterPass<ValueAccessTypes> W("val-access-types", "Get the access types of all values in a function", false, false);
