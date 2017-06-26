//===---------------------- recoverExpressions.cpp ------------------------===//
//
// This file is distributed under the Universidade Federal de Minas Gerais - 
// UFMG Open Source License. See LICENSE.TXT for details.
//
// Copyright (C) 2015   Gleison Souza Diniz Mendon?a
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include <fstream>
#include <queue>

#include "llvm/Analysis/RegionInfo.h"  
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/DIBuilder.h" 
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/ADT/Statistic.h"

#include "recoverExpressions.h"
#include "recoverFunctionCall.h"

#define JNON0 (j > 0) ? "," : "";

using namespace llvm;
using namespace std;
using namespace lge;

#define DEBUG_TYPE "recoverExpressions"
#define ERROR_VALUE -1

static cl::opt<bool> ClRegionTask("Region-Task",
cl::Hidden, cl::desc("Annotate regions in the source file."));

int RecoverExpressions::getIndex() {
  return this->index;
}

int RecoverExpressions::getNewIndex() {
  return (++(this->index));
}

void RecoverExpressions::addCommentToLine (std::string Comment,
                                         unsigned int Line) {     
  if (Comments.count(Line) == 0)
    Comments[Line] = Comment;
  else if (Comments[Line].find(Comment) == std::string::npos)
    Comments[Line] += Comment;
} 

void RecoverExpressions::copyComments (std::map <unsigned int, std::string>
                                      CommentsIn) {
  for (auto I = CommentsIn.begin(), E = CommentsIn.end(); I != E; ++I)
    addCommentToLine(I->second,I->first);
} 

int RecoverExpressions::getLineNo (Value *V) {
  if (!V)
    return ERROR_VALUE;
  if (Instruction *I = dyn_cast<Instruction>(V))
    if (I)
      if (MDNode *N = I->getMetadata("dbg"))
        if (N)
          if (DILocation *DI = dyn_cast<DILocation>(N))
            return DI->getLine();
  return ERROR_VALUE;
}

bool RecoverExpressions::isUniqueinLine(Instruction *I) {
  Function *F = I->getParent()->getParent();
  int line = getLineNo(I);
  for (auto B = F->begin(), BE = F->end(); B != BE; B++)
    for (auto II = B->begin(), IE = B->end(); II != IE; II++)
      if (isa<CallInst>(&(*II)))
        if ((line == getLineNo(II)) && ((&(*II)) != I)) {
          if (CallInst *CI = dyn_cast<CallInst>(II)) {
            Value *V = CI->getCalledValue();
            if (!isa<Function>(V))
              return false;
            Function *FF = cast<Function>(V);
            if (F->getName() == "llvm.dbg.declare")
              continue;
          }
          return false;
        }
  return true;
}

std::string RecoverExpressions::analyzeCallInst(CallInst *CI,
                                                const DataLayout *DT,
                                                RecoverCode *RC) {
  RecoverFunctionCall rfc;
  std::string computationName = "TM" + std::to_string(getIndex());
  rfc.setNAME(computationName);
  rfc.setRecoverNames(rn);
  rfc.initializeNewVars();

  rfc.annotataFunctionCall(CI, ptrRa, rp, aa, se, li, dt);
  Value *V = CI->getCalledValue();
  if (!isa<Function>(V))
    return std::string();
  Function *F = cast<Function>(V);
  std::string output = std::string();
  if (!F)
    return output;
  std::string name = F->getName();
  if ((F->isIntrinsic() || F->isDeclaration()) ||
      (name == "llvm.dbg.declare")) {
    valid = false;
    return output;
  }  
  // Define if this CALL INST is contained in the knowed tasks well
  // define by Task Miner
  bool isTask = false;
  this->liveIN.erase(this->liveIN.begin(), this->liveIN.end());
  this->liveOUT.erase(this->liveOUT.begin(), this->liveOUT.end());
  this->liveINOUT.erase(this->liveINOUT.begin(), this->liveINOUT.end());
  for (auto &I: this->tm->getTasks()) {
    I->print(errs());
    if (FunctionCallTask *FCT = dyn_cast<FunctionCallTask>(I)) {
      FCT->getFunctionCall()->dump();
      CI->dump();
      if (FCT->getFunctionCall() == CI) {
        this->liveIN = FCT->getLiveIN();
        this->liveOUT = FCT->getLiveOUT();
        this->liveINOUT = FCT->getLiveINOUT();
        isTask = true;
        break;
      }
    }
  }
  if (isTask == false) {
    errs() << "ERROR 6\n";
    return output;
  }
  if (CI->getNumArgOperands() == 0) {
    errs() << "ERROR 5\n";
    return "\n\n[UNDEF\nVALUE]\n\n";
  }
  output = std::string();
  std::map<Value*, std::string> strVal;
  for (unsigned int i = 0; i < CI->getNumArgOperands(); i++) {
    /*if (!isa<LoadInst>(CI->getArgOperand(i)) &&
        !isa<StoreInst>(CI->getArgOperand(i)) &&
        !isa<GetElementPtrInst>(CI->getArgOperand(i)) &&
        !isa<Argument>(CI->getArgOperand(i)) &&
        !isa<GlobalValue>(CI->getArgOperand(i)) &&
        !isa<AllocaInst>(CI->getArgOperand(i)))
      continue;
    }*/
     
    std::string str = analyzeValue(CI->getArgOperand(i), DT, RC);
    if (str == std::string()) {
      errs() << "ERROR 4:\n";
      CI->getArgOperand(i)->dump();
      return std::string();
    }
    strVal[CI->getArgOperand(i)] = str;
  }
  if (this->liveIN.size() != 0) {
    output += " depend(in:";
    bool isused = false;
    for (auto J = this->liveIN.begin(), JE = this->liveIN.end(); J != JE; J++) {
      if (isused)
        output += ",";
      isused = true;
      output += strVal[*J];
    }
    output += ")";
  }
  if (this->liveOUT.size() != 0) {
    output += " depend(out:";
    bool isused = false;
    for (auto J = this->liveOUT.begin(), JE = this->liveOUT.end(); J != JE; J++) {
      if (isused)
        output += ",";
      isused = true;
      output += strVal[*J];
    }
    output += ")";
  }
  if (this->liveINOUT.size() != 0) {
    output += " depend(inout:";
    bool isused = false;
    for (auto J = this->liveINOUT.begin(), JE = this->liveINOUT.end(); J != JE;
      J++) {
      if (isused)
        output += ",";
      isused = true;
      output += strVal[*J];
    } 
    output += ")";
  }
  return output;
}

std::string RecoverExpressions::analyzePointer(Value *V, const DataLayout *DT,
                                               RecoverCode *RC) {
    Instruction *I = cast<Instruction>(V);
    Value *BasePtrV = getPointerOperand(I);

    while (isa<LoadInst>(BasePtrV) || isa<GetElementPtrInst>(BasePtrV)) {
      if (LoadInst *LD = dyn_cast<LoadInst>(BasePtrV))
        BasePtrV = LD->getPointerOperand();

      if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(BasePtrV))
        BasePtrV = GEP->getPointerOperand();
    }

    int size = RC->getSizeToValue (BasePtrV, DT);
    int var = -1;
    std::string result = RC->getAccessString(V, "", &var, DT);
    if (result.find("[") == std::string::npos)
      return result;

    if (!RC->isValidPointer(BasePtrV, DT)) {
      valid = false;
      return std::string();
    }

    Type *tpy = V->getType();
    if ((tpy->getTypeID() == Type::HalfTyID) ||
        (tpy->getTypeID() == Type::FloatTyID) ||
        (tpy->getTypeID() == Type::DoubleTyID) ||
        (tpy->getTypeID() == Type::X86_FP80TyID) ||
        (tpy->getTypeID() == Type::FP128TyID) ||
        (tpy->getTypeID() == Type::PPC_FP128TyID) ||
        (tpy->getTypeID() == Type::IntegerTyID))
      return result;

    if (!RC->isValid())
      return std::string();
    
    size = RC->getSizeInBytes(size);
    std::string output = std::string();

    unsigned int i = 0;
    if (result.size() < 1)
      return std::string();

    if (result.find("[") == std::string::npos)
      return result;
    do {
      output += result[i];
      i++;
    } while((i < result.size()) && (result[i] != '['));

    if (result.size() > i)
      output += result[i];

    if (i > result.size()) {
      return std::string();
    }

    std::string newInfo = std::string();
    i++;

    bool needB = false;
    while((i < result.size()) && (result[i] != ']')) {
      if (result[i] == '[')
        needB = true;
      newInfo += result[i];
      i++;
    }
    if (needB)
      newInfo += result[i];   

    newInfo = "(" + newInfo + " / " + std::to_string(size) + ");\n";
    int var2 = -1;
    RC->insertCommand(&var2, newInfo);
    bool needEE = (output.find("[") != std::string::npos);
    output = output + "TM" + std::to_string(getIndex());
    output += "[" + std::to_string(var2) + "]";
    if (needEE)
      output += "]";
    return output;

}

std::string RecoverExpressions::analyzeValue(Value *V, const DataLayout *DT,
RecoverCode *RC) {
  if (valid == false) {
    return std::string();
  }
  else if (CallInst *CI = dyn_cast<CallInst>(V)) {
    return analyzeCallInst(CI, DT, RC);
  }
  else if ((isa<StoreInst>(V) || isa<LoadInst>(V)) ||
           isa<GetElementPtrInst>(V)) {
    return analyzePointer(V, DT, RC);
  }
  else {
    int var = -1;
    std::string result = RC->getAccessString(V, "", &var, DT);
    if (!RC->isValid())
      return std::string();
    if (result == std::string())
      return ("TM" + std::to_string(getIndex()) + "["+ std::to_string(var)) + "]";
    return result;
  }
}

void RecoverExpressions::annotateExternalLoop(Instruction *I) {
  Region *R = rp->getRegionInfo().getRegionFor(I->getParent());
  Loop *LL = this->li->getLoopFor(I->getParent());
  if (LL) {
    if (LL->getHeader())
      R = rp->getRegionInfo().getRegionFor(LL->getHeader());
  }
  /*if (!st->isSafetlyRegionLoops(R)) {
    return;
  }*/
  Loop *L = this->li->getLoopFor(I->getParent());
  int line = st->getStartRegionLoops(R).first;
  errs() << "Writing in the line = " << std::to_string(line) << "\n";
  std::string output = std::string();
  output += "#pragma omp parallel\n#pragma omp single\n";
  addCommentToLine(output, line);
}

void RecoverExpressions::analyzeFunction(Function *F) {
  const DataLayout DT = F->getParent()->getDataLayout();
  RecoverCode RC;

  std::string computationName = "TM" + std::to_string(getIndex());
  RC.setNAME(computationName);
  RC.setRecoverNames(rn);
  RC.initializeNewVars();

  std::map<Loop*, bool> loops;
  for (auto BB = F->begin(), BE = F->end(); BB != BE; BB++) {
    for (auto I = BB->begin(), IE = BB->end(); I != IE; I++) {
      if (isa<CallInst>(I)) {
        valid = true;
        computationName = "TM" + std::to_string(getNewIndex());
        RC.setNAME(computationName);
        std::string result = analyzeValue(I, &DT, &RC);
        if (result != std::string()) {
          std::string output = std::string();
          if (RC.getIndex() > 0) {
            output += "long long int " + computationName + "[";
            output += std::to_string(RC.getNewIndex()) + "];\n";
            output += RC.getUniqueString();
            RC.clearCommands();
            if (!RC.isValid()) {
               errs() << "ERROR 3\n";
               continue;
            }
          }
          if (result != "\n\n[UNDEF\nVALUE]\n\n")
            output += "#pragma omp task" + result + "\n";
          else
            output += "#pragma omp task\n";
          Region *R = rp->getRegionInfo().getRegionFor(BB);
          int line = getLineNo(I);
          Loop *L = this->li->getLoopFor(I->getParent());
          if (!isUniqueinLine(I)) {
            errs() << "ERROR 1\n";
            continue;
          }
          
          if ((loops.count(L) == 0) && st->isSafetlyRegionLoops(R)) {
            annotateExternalLoop(I);
            loops[L] = true;
          }
          if(st->isSafetlyRegionLoops(R))
            addCommentToLine(output, line);
          else {
            errs() << "ERROR 2\n";
            R->dump(); 
          }
        }
      }
    }
  }
}

Value *RecoverExpressions::getPointerOperand(Instruction *Inst) {
  if (LoadInst *Load = dyn_cast<LoadInst>(Inst))
    return Load->getPointerOperand();
  else if (StoreInst *Store = dyn_cast<StoreInst>(Inst))
    return Store->getPointerOperand();
  else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Inst))
    return GEP->getPointerOperand();

  return 0;
}

std::string RecoverExpressions::extractDataPragma(Region *R) {
  std::string pragma = std::string();
  bool hasLOAD = false;
  bool hasLOADSTORE = false;
  std::map<std::string, char> pointers;
  const DataLayout DT = R->block_begin()->getParent()->getParent()->getDataLayout();

  RecoverCode RC;
  std::string computationName = "TM" + std::to_string(getNewIndex());
  RC.setNAME(computationName);
  RC.setRecoverNames(rn);
  RC.initializeNewVars();

  for (BasicBlock *BB : R->blocks())
    for (auto I = BB->begin(), E = --BB->end(); I != E; ++I) {
      if (isa<LoadInst>(I) || isa<StoreInst>(I) || isa<GetElementPtrInst>(I)) {
        Value *BasePtrV = getPointerOperand(I);
        while (isa<LoadInst>(BasePtrV) || isa<GetElementPtrInst>(BasePtrV)) {
          if (LoadInst *LD = dyn_cast<LoadInst>(BasePtrV))
            BasePtrV = LD->getPointerOperand();
          if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(BasePtrV))
            BasePtrV = GEP->getPointerOperand();
        }
        int var = -1;
        std::string name = RC.getAccessString(BasePtrV, "", &var, &DT);
        pointers[name] = this->ptrRa->getPointerAcessType(R, BasePtrV);
        if (pointers[name] == 1)
          hasLOAD = true;
        if (pointers[name] == 3)
          hasLOADSTORE = true;
      }
    }

  pragma += "#pragma omp parallel\n#pragma omp single\n";
  if (hasLOAD || hasLOADSTORE)
    pragma += "#pragma omp task depend(";

  if (hasLOAD) {
    pragma += "in:";
    bool hasV = false;
    for (auto I = pointers.begin(), IE = pointers.end(); I != IE; I++) {
      if (I->second == 1) {
        if (hasV == true)
          pragma += ",";
        hasV = true;
        pragma += I->first;
      }
    }
  }

  if (hasLOADSTORE) {
    if (hasLOAD)
      pragma += ",";
    pragma += "inout:";
    bool hasV = false;
    for (auto I = pointers.begin(), IE = pointers.end(); I != IE; I++) {
      if (I->second == 3) {
        if (hasV == true)
          pragma += ",";
        hasV = true;
        pragma += I->first;
      }
    }
  }

  if(hasLOAD || hasLOADSTORE)
    pragma += ")\n{\n";

  return pragma; 
}

void RecoverExpressions::analyzeRegion(Region *R) {
  int line = st->getStartRegionLoops(R).first;
  int lineEnd = st->getEndRegionLoops(R).first + 1;
  if (!st->isSafetlyRegionLoops(R)) {
    for (auto SR = R->begin(), SRE = R->end(); SR != SRE; ++SR)
      analyzeRegion(&(**SR));
    return;
  }
  std::string output = std::string();
  output += extractDataPragma(R);
  addCommentToLine(output, line);
  output = "}\n";
  addCommentToLine(output, lineEnd);
}

bool RecoverExpressions::runOnFunction(Function &F) {
  this->li = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  this->rp = &getAnalysis<RegionInfoPass>();
  this->aa = &getAnalysis<AliasAnalysis>();
  this->se = &getAnalysis<ScalarEvolution>();
  this->dt = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  this->rn = &getAnalysis<RecoverNames>();
  this->rr = &getAnalysis<RegionReconstructor>();
  this->st = &getAnalysis<ScopeTree>();
  this->ptrRa = &getAnalysis<PtrRangeAnalysis>();
  this->tm = &getAnalysis<TaskMiner>();

  index = 0;
  if (ClRegionTask == true)
    analyzeRegion(this->rp->getRegionInfo().getTopLevelRegion());   
  else
    analyzeFunction(&F);

  return true;
}

char RecoverExpressions::ID = 0;
static RegisterPass<RecoverExpressions> Z("recoverExpressions",
"Recover Expressions to the source File.");

//===------------------------ recoverExpressions.cpp ------------------------===//
