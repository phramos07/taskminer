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

using namespace llvm;
using namespace std;
using namespace lge;

#define DEBUG_TYPE "recoverExpressions"
#define ERROR_VALUE -1

void RecoverExpressions::setTasksList(std::list<Task*> taskList) {
  this->tasksList = taskList;
} 

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

void RecoverExpressions::findRecursiveTasks() {
  for (auto &I: this->tasksList) { 
    if (RecursiveTask *RT = dyn_cast<RecursiveTask>(I)) { 
      isRecursive[RT->getRecursiveCall()->getCalledFunction()] = true;
      isFCall[RT->getRecursiveCall()] = true;
    }
    if (FunctionCallTask *FCT = dyn_cast<FunctionCallTask>(I)) {
      isFCall[FCT->getFunctionCall()] = true;
    }  
  }
}

void RecoverExpressions::insertCutoff(Function *F) {
  int start = st->getMinLineFunction(F) + 1;
  int end = st->getMaxLineFunction(F);
  addCommentToLine("taskminer_depth_cutoff++;\n", start);
  addCommentToLine("taskminer_depth_cutoff--;\n", end);
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
                                                RecoverPointerMD *RPM) {
  /*RecoverFunctionCall rfc;
  std::string computationName = "TM" + std::to_string(getIndex());
  rfc.setNAME(computationName);
  rfc.setRecoverNames(rn);
  rfc.initializeNewVars();
  
  rfc.annotataFunctionCall(CI, ptrRa, rp, aa, se, li, dt);*/
  Value *V = CI->getCalledValue();
  
  if (!isa<Function>(V)) { 
    return std::string();
  }
  Function *F = cast<Function>(V);
  std::string output = std::string();
  if (!F) {
    return output;
  }
  std::string name = F->getName();
  /*if (F->empty() || (name == "llvm.dbg.declare")) {
    valid = false;
    return output;
  } */
  // Define if this CALL INST is contained in the knowed tasks well
  // define by Task Miner
  bool isTask = false;
  bool hasTaskWait = false;
  bool isInsideLoop = true;
  Loop *L1 = nullptr;
  Loop *L2 = nullptr;
  this->liveIN.erase(this->liveIN.begin(), this->liveIN.end());
  this->liveOUT.erase(this->liveOUT.begin(), this->liveOUT.end());
  this->liveINOUT.erase(this->liveINOUT.begin(), this->liveINOUT.end());
  for (auto &I: this->tasksList) {
//    if (!(I->getCost().aboveThreshold()))
//      continue;
    if (FunctionCallTask *FCT = dyn_cast<FunctionCallTask>(I)) {
      if (FCT->getFunctionCall() == CI) {
        this->liveIN = FCT->getLiveIN();
        this->liveOUT = FCT->getLiveOUT();
        this->liveINOUT = FCT->getLiveINOUT();
        if (FCT->hasSyncBarrier()) {
          L1 = this->li->getLoopFor(CI->getParent());
          L2 = L1;
          while (L2->getParentLoop()) {
            L2 = L2->getParentLoop();
          }
        }
        isTask = true;
        break;
      }
    }

    if (RecursiveTask *RT = dyn_cast<RecursiveTask>(I)) {

      if (RT->getRecursiveCall() == CI) {
        this->liveIN = RT->getLiveIN();
        this->liveOUT = RT->getLiveOUT();
        this->liveINOUT = RT->getLiveINOUT();
        isTask = true;
        hasTaskWait = RT->hasSyncBarrier();
        isInsideLoop = RT->insideLoop();
        insertCutoff(CI->getCalledFunction() );
        /*if (RT->hasSyncBarrier()) {
          L1 = this->li->getLoopFor(CI->getParent());
          L2 = L1;
          if (L2)
            while (L2->getParentLoop()) {
              L2 = L2->getParentLoop();
          }
        }*/
        break;
      }
    }
  }
  if (isTask == false) {
    return output;
  }
  errs() << "Printing\n\n\n";
  CI->dump();
  if (L1)
    L1->dump();
  errs() << "Parent\n\n";
  if(L2)
    L2->dump();
  errs() << "Annotating:\n";
  CI->dump();
  if (CI->getNumArgOperands() == 0) {
    return "\n\n[UNDEF\nVALUE]\n\n";
  }
  output = std::string();
  std::map<Value*, std::string> strVal;
  for (unsigned int i = 0; i < CI->getNumArgOperands(); i++) {
    if (!isa<LoadInst>(CI->getArgOperand(i)) &&
        !isa<StoreInst>(CI->getArgOperand(i)) &&
        !isa<GetElementPtrInst>(CI->getArgOperand(i)) &&
        !isa<Argument>(CI->getArgOperand(i)) &&
        !isa<GlobalValue>(CI->getArgOperand(i)) &&
        !isa<AllocaInst>(CI->getArgOperand(i)) &&
        !isa<BitCastInst>(CI->getArgOperand(i))) {
      continue;
    }
     
    std::string str = analyzeValue(CI->getArgOperand(i), DT, RPM);
    if (str == std::string() || str == "0") {
      errs() << "Error: CANNOT READ INSTRUCTION: ";
      CI->getArgOperand(i)->dump();
      return std::string();
    }
    strVal[CI->getArgOperand(i)] = str;
  }
  errs() << "YEEEEESSSSSSSSSSSSSS!!!!!!!!!!\n";
  annotateExternalLoop(CI, L1, L2);
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
  if (hasTaskWait) {
    std::string wait = "#pragma omp taskwait\n";
    int line = getLineNo(CI);
    errs() << "TASK WAIT:\n";
    CI->dump();
    line++;
    addCommentToLine(wait, line);  
  }
  // HERE
  if (output == std::string()) {
    output = "\n\n[UNDEF\nVALUE]\n\n";
  }
  return output;
}

std::string RecoverExpressions::analyzePointer(Value *V, const DataLayout *DT,
                                               RecoverPointerMD *RPM) {
    Instruction *I = cast<Instruction>(V);
    Value *BasePtrV = getPointerOperand(I);

    while (isa<LoadInst>(BasePtrV) || isa<GetElementPtrInst>(BasePtrV)) {
      if (LoadInst *LD = dyn_cast<LoadInst>(BasePtrV))
        BasePtrV = LD->getPointerOperand();

      if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(BasePtrV))
        BasePtrV = GEP->getPointerOperand();
    }

    int size = RPM->getSizeToValue (BasePtrV, DT);
    int var = -1;
    std::string result = RPM->getAccessStringMD(V, "", &var, DT);
    if (result.find("[") == std::string::npos) {
      return result;
    }

    if (!RPM->isValidPointer(BasePtrV, DT)) {
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
        (tpy->getTypeID() == Type::IntegerTyID)) {
      return result;
    }

    if (!RPM->isValid()) {
      return std::string();
    }
    
    size = RPM->getSizeInBytes(size);
    std::string output = std::string();

    unsigned int i = 0;
    if (result.size() < 1) {
      return std::string();
    }

    if (result.find("[") == std::string::npos) {
      return result;
    }
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
    RPM->insertCommand(&var2, newInfo);
    bool needEE = (output.find("[") != std::string::npos);
    output = output + "TM" + std::to_string(getIndex());
    output += "[" + std::to_string(var2) + "]";
    if (needEE)
      output += "]";
    return output;

}

std::string RecoverExpressions::analyzeValue(Value *V, const DataLayout *DT,
RecoverPointerMD *RPM) {
  if (valid == false) {
    return std::string();
  }
  else if (CallInst *CI = dyn_cast<CallInst>(V)) {
    return analyzeCallInst(CI, DT, RPM);
  }
/*  else if ((isa<StoreInst>(V) || isa<LoadInst>(V)) ||
           isa<GetElementPtrInst>(V)) {
    return analyzePointer(V, DT, RPM);
  }*/
  else {
    int var = -1;
    std::string result = RPM->getAccessStringMD(V, "", &var, DT);
    if (!RPM->isValid())
      return std::string();
    if (result == std::string())
      return ("TM" + std::to_string(getIndex()) + "["+ std::to_string(var)) + "]";
    return result;
  }
}

void RecoverExpressions::annotateExternalLoop(Instruction *I, Loop *L1,
                                              Loop *L2) {
  /*if (!L1 && !L2) {
    annotateExternalLoop(I);
    return;
  }*/
  if (L2) {
    Region *R = rp->getRegionInfo().getRegionFor(I->getParent());
    if (L2->getHeader())
      R = rp->getRegionInfo().getRegionFor(L2->getHeader()); 
    if (!st->isSafetlyRegionLoops(R))
      return;
    int line = st->getStartRegionLoops(R).first;  
    std::string output = std::string();
    output += "#pragma omp parallel\n#pragma omp single\n";
    addCommentToLine(output, line);
  }  
  if (L1) {
    Region *R = rp->getRegionInfo().getRegionFor(I->getParent());
    if (L1->getHeader())
      R = rp->getRegionInfo().getRegionFor(L1->getHeader()); 
    if (!st->isSafetlyRegionLoops(R))
      return;
    int line = st->getEndRegionLoops(R).first;  
    line++;
    std::string output = std::string();
    output += "#pragma omp taskwait\n";
    addCommentToLine(output, line);
  }
}

void RecoverExpressions::annotateExternalLoop(Instruction *I) {
  Loop *LL = this->li->getLoopFor(I->getParent());
  Region *R = rp->getRegionInfo().getRegionFor(I->getParent()); 
  if (LL) {
    if (LL->getHeader())
      R = rp->getRegionInfo().getRegionFor(LL->getHeader());
  }
  if (!st->isSafetlyRegionLoops(R)) {
    return;
  }
  Loop *L = this->li->getLoopFor(I->getParent());
  int line = st->getStartRegionLoops(R).first;
  std::string output = std::string();
  output += "#pragma omp parallel\n#pragma omp single\n";
  addCommentToLine(output, line);
}

void RecoverExpressions::analyzeFunction(Function *F) {
  const DataLayout DT = F->getParent()->getDataLayout();

  std::map<Loop*, bool> loops;
  for (auto BB = F->begin(), BE = F->end(); BB != BE; BB++) {
    for (auto I = BB->begin(), IE = BB->end(); I != IE; I++) {
      if (isa<CallInst>(I)) {
        valid = true;
        RecoverPointerMD RPM;
        std::string computationName = "TM" + std::to_string(getNewIndex());
        RPM.setNAME(computationName);
        RPM.setRecoverNames(rn);
         RPM.initializeNewVars();
        std::string result = analyzeValue(I, &DT, &RPM);
        std::string check = std::string();
        if (result != std::string()) {
          std::string output = std::string();
          if (RPM.getIndex() > 0) {
            output += "long long int " + computationName + "[";
            output += std::to_string(RPM.getNewIndex()) + "];\n";
            output += RPM.getUniqueString();
            RPM.clearCommands();
            if (!RPM.isValid()) {
               continue;
            }
          }
          output += "cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);\n";
          if (isRecursive.count(I->getParent()->getParent()) > 0)
            check = " if(cutoff_test)";
          if (result != "\n\n[UNDEF\nVALUE]\n\n") {
            output += "#pragma omp task untied default(shared)" + result;
            output += check + "\n";
          }
          else
            output += "#pragma omp task untied default(shared)" + check + "\n";
          Region *R = rp->getRegionInfo().getRegionFor(BB);
          int line = getLineNo(I);
          /*Loop *L = this->li->getLoopFor(I->getParent());
          if (!isUniqueinLine(I)) {
            errs() << "Bug 1\n";
            continue;
          }
          if ((loops.count(L) == 0)) { //&& st->isSafetlyRegionLoops(R)) {
            annotateExternalLoop(I);
            loops[L] = true;
          }*/
          //if(st->isSafetlyRegionLoops(R)) {
            addCommentToLine(output, line);
          //}
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
  this->liveIN.erase(this->liveIN.begin(), this->liveIN.end());
  this->liveOUT.erase(this->liveOUT.begin(), this->liveOUT.end());
  this->liveINOUT.erase(this->liveINOUT.begin(), this->liveINOUT.end());

  RegionTask *RT = bbsRegion[R];
  this->liveIN = RT->getLiveIN();
  this->liveOUT = RT->getLiveOUT();
  this->liveINOUT = RT->getLiveINOUT();  
  
  RT->print(errs());

  std::string pragma = std::string();
  std::string output = std::string();
  const DataLayout DT = R->block_begin()->getParent()->getParent()->getDataLayout();

  RecoverPointerMD RPM;
  std::string computationName = "TM" + std::to_string(getNewIndex());
  RPM.setNAME(computationName);
  RPM.setRecoverNames(rn);
  RPM.initializeNewVars();
  valid = true;

  if (this->liveIN.size() != 0) {
    output += " depend(in:";
    bool isused = false;
    for (auto J = this->liveIN.begin(), JE = this->liveIN.end(); J != JE; J++) {
      std::string str = analyzeValue(*J, &DT, &RPM);
      if (isused)
        output += ",";
      isused = true;
      output += str;
    }
    output += ")";
  }

  if (this->liveOUT.size() != 0) {
    output += " depend(out:";
    bool isused = false;
    for (auto J = this->liveOUT.begin(), JE = this->liveOUT.end(); J != JE; J++) {
      std::string str = analyzeValue(*J, &DT, &RPM); 
      if (isused)
        output += ",";
      isused = true;
      output += str;
    }
    output += ")";
  }
  if (this->liveINOUT.size() != 0) {
    output += " depend(inout:";
    bool isused = false;
    for (auto J = this->liveINOUT.begin(), JE = this->liveINOUT.end(); J != JE;
      J++) {
      std::string str = analyzeValue(*J, &DT, &RPM);
      if (isused)
        output += ",";
      isused = true;
      output += str;
    } 
    output += ")";
  }
 
  if (RPM.getIndex() > 0) {
    pragma += "long long int " + computationName + "[";
    pragma += std::to_string(RPM.getNewIndex()) + "];\n";
    pragma += RPM.getUniqueString();
    RPM.clearCommands();
  }
  pragma += "#pragma omp task untied default(shared) " + output + "\n{\n";
  return pragma; 
}

void RecoverExpressions::analyzeRegion(Region *R) {
  if (!R)
    return;
  if (bbsRegion.count(R)) {
    std::set<BasicBlock*> BBS = bbsRegion[R]->getbbs();
    int line = st->getMinLine(BBS);
    int lineEnd = st->getMaxLine(BBS) + 1;
    if (!st->isSafetlyInstSet(BBS)) {
      if (!st->getSmallestScope(BBS, &line, &lineEnd)) {
//     errs() << "Start : " << line << " = " << lineEnd << "\n";
      for (auto SR = R->begin(), SRE = R->end(); SR != SRE; ++SR)
        analyzeRegion(&(**SR));
        return;
      }
    }
    annotateExternalLoop((*BBS.begin())->begin());
    std::string output = std::string();
    output += extractDataPragma(R);
    addCommentToLine(output, line);
    output = "}\n";
    addCommentToLine(output, lineEnd);
  }
  for (auto SR = R->begin(), SRE = R->end(); SR != SRE; ++SR)
    analyzeRegion(&(**SR));
}

/*bool RecoverCode::analyzeLoop (Loop* L, int Line, int LastLine,
                                        PtrRangeAnalysis *ptrRA, 
                                        RegionInfoPass *rp, AliasAnalysis *aa,
                                        ScalarEvolution *se, LoopInfo *li,
                                        DominatorTree *dt, std::string & test) {
  
  // Initilize The Analisys with Default Values.
  initializeNewVars(); 

  Module *M = L->getLoopPredecessor()->getParent()->getParent();
  const DataLayout DT = DataLayout(M);
  std::map<Value*, std::pair<Value*, Value*> > pointerBounds;
  std::string expression = std::string();
  std::string expressionEnd = std::string();

  Restrictifier Rst = Restrictifier();
  Rst.setAliasAnalysis(aa);
  Region *r = regionofBasicBlock((L->getLoopPreheader()), rp);

  if (!ptrRA->RegionsRangeData[r].HasFullSideEffectInfo)
    r = regionofBasicBlock((L->getHeader()), rp);
 
  if (!ptrRA->RegionsRangeData[r].HasFullSideEffectInfo)
    return false;
    
  Instruction *insertPt = r->getEntry()->getTerminator();
  SCEVRangeBuilder rangeBuilder(se, DT, aa, li, dt, r, insertPt);

  // Generate and store both bounds for each base pointer in the region.
  for (auto& pair : ptrRA->RegionsRangeData[r].BasePtrsData) {
    if (pointerDclInsideLoop(L,pair.first))
      continue;
    // Adds "sizeof(element)" to the upper bound of a pointer, so it gives us
    // the address of the first byte after the memory region.
    Value *low = rangeBuilder.getULowerBound(pair.second.AccessFunctions);
    Value *up = rangeBuilder.getUUpperBound(pair.second.AccessFunctions);
    up = rangeBuilder.stretchPtrUpperBound(pair.first, up);
    pointerBounds.insert(std::make_pair(pair.first, std::make_pair(low, up)));
    }

  std::map<std::string, std::string> vctLower;
  std::map<std::string, std::string> vctUpper;
  std::map<std::string, char> vctPtMA;
  std::map<std::string, Value*> vctPtr;
  std::map<std::string, bool> needR;

  for (auto It = pointerBounds.begin(), EIt = pointerBounds.end(); It != EIt;
       ++It) {
    
    RecoverNames::VarNames nameF = rn->getNameofValue(It->first);
    Rst.setNameToValue(nameF.nameInFile, It->first);
    std::string lLimit = getAccessExpression(It->first, It->second.first,
        &DT, false);
    std::string uLimit = getAccessExpression(It->first, It->second.second,
        &DT, true);
   
    std::string olLimit = std::string();
    std::string oSize = std::string();
    generateCorrectUB(lLimit, uLimit, olLimit, oSize);
    vctLower[nameF.nameInFile] = olLimit;
    vctUpper[nameF.nameInFile] = oSize;
    vctPtMA[nameF.nameInFile] = ptrRA->getPointerAcessType(L, It->first);
    vctPtr[nameF.nameInFile] = It->first;
    needR[nameF.nameInFile] = needPointerAddrToRestrict(It->first);
    if (!isValid()) {
      errs() << "[TRANSFER-PRAGMA-INSERTION] WARNING: unable to generate C " <<
        " code for bounds of pointer: " << (nameF.nameInFile.empty() ?
        "<unable to recover pointer name>" : nameF.nameInFile) << "\n";
      return isValid();
    }

  }
  
  expression += getDataPragma(vctLower, vctUpper, vctPtMA);

  if (isValid()) {
    std::string result = std::string(); 
    if (getIndex() > 0) {
      result += "long long int " + NAME + "[";
      result += std::to_string(getNewIndex()) + "];\n";
      result += getUniqueString();
    }
    result += expression;

    if (OMPF == OMP_GPU)
      Rst.setTrueOMP();

    Rst.setName("RST_"+NAME);
    Rst.getBounds(vctLower, vctUpper, vctPtr, needR);
    result = Rst.generateTests(result);

    restric = Rst.isValid();
    // Use to insert test on parallel pragmas
    //if (Rst.isValid())
    //  test = "if(!RST_" + NAME + ")"; 
    Comments[Line] = result;
  }
  return isValid();
}
*/
void RecoverExpressions::getRegionFromRegionTask(RegionTask *RT) {
  Region *R;
  std::set<BasicBlock*> bbs = RT->getbbs();
  if (!(*bbs.begin())) {
    return;
  }
  R = this->rp->getRegionInfo().getRegionFor(*(bbs.begin()));
  if (!R) {
    return;
  }
  for (auto BB : bbs) {
    Region *RR = this->rp->getRegionInfo().getRegionFor(BB);
    if (RR != nullptr)
      R = this->rp->getRegionInfo().getCommonRegion(RR, R);
  }
  if (R)
    bbsRegion[R] = RT;
}

void RecoverExpressions::getTaskRegions() {
  bbsRegion.erase(bbsRegion.begin(), bbsRegion.end());
  for (auto &I: this->tasksList) {
//    if (!(I->getCost().aboveThreshold()))
//      continue;
    if (RegionTask *RT = dyn_cast<RegionTask>(I)) {
        getRegionFromRegionTask(RT);
    }
  }
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

  findRecursiveTasks();
  
  std::string header = "#include <omp.h>\n";
  header += "#ifndef taskminerutils\n";
  header += "#define taskminerutils\n"; 
  header += "static int taskminer_depth_cutoff = 0;\n";
  header += "#define DEPTH_CUTOFF omp_get_num_threads()\n";
  header += "char cutoff_test = 0;\n";
  header += "#endif\n"; 
  addCommentToLine(header, 1);
  index = 0;
  analyzeFunction(&F);
  getTaskRegions();
  analyzeRegion(this->rp->getRegionInfo().getTopLevelRegion());   

  return true;
}

char RecoverExpressions::ID = 0;
static RegisterPass<RecoverExpressions> Z("recoverExpressions",
"Recover Expressions to the source File.");

//===------------------------ recoverExpressions.cpp ------------------------===//
