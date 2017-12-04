//===---------------------- recoverExpressions.h --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the Universidade Federal de Minas Gerais -
// UFMG Open Source License. See LICENSE.TXT for details.
//
// Copyright (C) 2015   Gleison Souza Diniz Mendon?a
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/LoopInfo.h"

#include "../TaskMiner.h"
#include "extractSourceData.h" 

#ifndef myutils2
#define myutils2

#include "recoverPointerMD.h"
#endif

using namespace lge;

namespace llvm {
class ScalarEvolution;
class AliasAnalysis;
class SCEV;
class DominatorTree;

class DominanceFrontier;
struct PostDominatorTree;
class Value;
class Region;
class Instruction;
class LoopInfo;
class ArrayInference;
class ScopeTree;

class RecoverExpressions : public FunctionPass {

  private:

  //===---------------------------------------------------------------------===
  //                              Data Structs
  //===---------------------------------------------------------------------===
  bool valid;

  std::string NAME;

  int index;

  std::set<Value*> liveIN;
  std::set<Value*> liveOUT;
  std::set<Value*> liveINOUT;

  ExtractSourceData esd;

  std::map<Region*, RegionTask*> bbsRegion;

  std::map<Function*, bool> isRecursive;

  std::map<CallInst*, bool> isFCall;

  std::set<CallInst*> tasksCalls;

  //===---------------------------------------------------------------------===

  // Methods to manage the correct computation auxiliar names.
  int getIndex();
  int getNewIndex();

  // Identify all Recursive Tasks.
  void findRecursiveTasks();

  // Return the last Branch for a function F.
  int getLastBranchLine(Function *F);

  // Insert the Cutoff in recursive functions.
  void insertCutoff(Function *F);

  // Analyze a call instruction.
  std::string analyzeCallInst(CallInst *CI, const DataLayout *DT,
                              RecoverPointerMD *RPM);

  // Analyze Instructions that use memory.
  std::string analyzePointer(Value *V, const DataLayout *DT, 
                             RecoverPointerMD *RPM);
 
  // Analyze a value present in the IR.
  std::string analyzeValue(Value *V, const DataLayout *DT, RecoverPointerMD *RPM);

  // Find and delimitate the expressions in a function.
  void analyzeFunction(Function *F);

  // Annotate the pragmas before a loop, case necessary.
  void annotateExternalLoop(Instruction *I);
  void annotateExternalLoop(Instruction *I, Loop *L1, Loop *L2);

  // Return the line number for Value V.
  int getLineNo (Value *V);

  // Verify if the source file have just the line of called instruction and
  // its dependences.
  /// Just work with CallInst.
  bool isUniqueinLine(Instruction *I);

  // Copies the resulting Comments to the map CommentsIn.
  void copyComments (std::map<unsigned int, std::string> CommentsIn);

  // Adds provided comments into line in original file.
  void addCommentToLine (std::string Comment, unsigned int Line);
  void addCommentToBLine (std::string Comment, unsigned int Line);

  // Analyze and annotate regions.
  void analyzeRegion(Region *R);

  // Extract a pragma with ( in / out ) data transference. 
  std::string extractDataPragma(Region *R);

  // Return the value with the pointer operand.
  Value *getPointerOperand(Instruction *Inst);

  // Set the task regions
  void getTaskRegions();

  public:

  //===---------------------------------------------------------------------===
  //                              Data Structs
  //===---------------------------------------------------------------------===
   std::map<unsigned int, std::string> Comments;
  //===---------------------------------------------------------------------===

  static char ID;

  RecoverExpressions() : FunctionPass(ID) {};
 
  void setTasksList(std::list<Task*> taskList);

  void setTasksCalls(std::set<CallInst*> taskCalls);

  void getRegionFromRegionTask(RegionTask *RT);
 
  // We need to insert the Instructions for each source file.
  virtual bool runOnFunction(Function &F) override;

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<RegionInfoPass>();
      AU.addRequired<AliasAnalysis>();
      AU.addRequired<ScalarEvolution>();
      AU.addRequiredTransitive<LoopInfoWrapperPass>();
      AU.addRequired<RecoverNames>();
      AU.addRequired<DominatorTreeWrapperPass>();
      AU.addRequired<RegionReconstructor>(); 
      AU.addRequired<ScopeTree>();
      AU.addRequired<PtrRangeAnalysis>();
      AU.setPreservesAll();
  }

  RecoverNames *rn;
  RegionInfoPass *rp;
  AliasAnalysis *aa;
  ScalarEvolution *se;
  LoopInfo *li;
  DominatorTree *dt;
  RegionReconstructor *rr;
  ScopeTree *st;
  PtrRangeAnalysis *ptrRa;
  std::list<Task*> tasksList;
};

}

//===---------------------- recoverExpressions.h --------------------------===//
