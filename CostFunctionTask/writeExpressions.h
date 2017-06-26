//===------------------------ writeExpressions.h --------------------------===//
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
// 
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/LoopInfo.h"

#ifndef myutils
#define myutils
#include "recoverCode.h"
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

class WriteExpressions : public FunctionPass {

  private:

  //===---------------------------------------------------------------------===
  //                              Data Structs
  //===---------------------------------------------------------------------===
  unsigned int NewVars;

  std::string VETNAME = "LLVM";

  std::vector<std::string> Expression;
  
  std::map<Loop*, bool> isknowedLoop;
  //===---------------------------------------------------------------------===

  // Find the lines to parallelize in standard input file.
  void readParallelLoops ();

  // Analyze loops and count valid call instructions inside them.
  void analyzeCalls (Loop *L);

  // Return the line number for Value V.
  int getLineNo (Value *V);
  
  // Clear the vector Expression.
  void clearExpression ();

  // Return the vector Expression in one simple string.
  std::string getUniqueString();

  // This void inserts a new Region into loopMap.
  void insertRegion (Loop *L, Region *R);

  // Returns true for analyzable regions, false otherwise.
  bool analyzeLoop (Loop* L, int Line);

  // Provides extra debug information, i.e. the last line of the loop L.
  int returnLoopEndLine (Loop *L);

  // Provides extra debug information, i.e. the last line of the region R.
  int returnRegionEndLine (Region *R);

  // Provides extra debug information, i.e. the first line of the region R
  int returnRegionStartLine (Region *R);

  // Generate statistics to analyzed loops.
  void marknumAL (Loop *L);

  // Generate statistics to annotated loops.
  void marknumWL (Loop *L);

  // Search for every sub region in region R.
  void regionIdentify(Region *R);
 
  // Annotate pragma 'Kernels' in the region R
  bool annotateAccKernels (Region *R, std::string NAME, bool restric);

  // write the pragma 'Kernels' in association with loop L
    void writeKernels (Loop *L, std::string NAME, bool restric);

  // Identify the region case it is safe to do memory coalescing.
  bool isSafeMemoryCoalescing (Region *R);
 
  // Search for every sub region in region R, trying to identify the best region
  // to agrupate memory data transferences..
  void regionIdentifyCoalescing(Region *R);

  // This void calls regionIdentify for the top level region in function F.
  void functionIdentify(Function *F);

  // Use the metadata to validate insertion of "loop independent" pragmas
  void denotateLoopParallel (Loop *L, std::string condition);

  // Return true if the loop "L" has isParallel metadata, and false case not.
  bool isLoopParallel (Loop *L);

  // Returns true if the region R has any loop annotated as parallel. 
  bool hasLoopParallel (Region *R);

  // Returns true if the loop L is Analyzable.
  bool isLoopAnalyzable(Loop *L);

  // Returns the region of the Basic Block.
  Region* regionofBasicBlock(BasicBlock *bb);

  // Copies the resulting Comments to the map CommentsIn.
  void copyComments (std::map<unsigned int, std::string> CommentsIn);

  // Adds provided comments into line in original file.
  void addCommentToLine (std::string Comment, unsigned int Line);

  // For Region "R", write the correct computation, bounds and parallel
  // annotations.
  void writeComputation (int line, int lineEnd, Region *R);
  
  public:

  //===---------------------------------------------------------------------===
  //                              Data Structs
  //===---------------------------------------------------------------------===
  std::map<unsigned int, std::string> Comments;
  //===---------------------------------------------------------------------===

  static char ID;

  WriteExpressions() : FunctionPass(ID) {};
  
  // We need to insert the Instructions for each source file.
  virtual bool runOnFunction(Function &F) override;

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<RegionInfoPass>();
      AU.addRequired<AliasAnalysis>();
      AU.addRequired<ScalarEvolution>();
      AU.addRequiredTransitive<LoopInfoWrapperPass>();
      AU.addRequired<PtrRangeAnalysis>();
      AU.addRequired<RecoverNames>();
      AU.addRequired<DominatorTreeWrapperPass>();
      AU.addRequired<RegionReconstructor>(); 
      AU.addRequired<ScopeTree>();
      AU.setPreservesAll();
  }

  RecoverNames *rn;
  PtrRangeAnalysis *ptrRA;
  RegionInfoPass *rp;
  AliasAnalysis *aa;
  ScalarEvolution *se;
  LoopInfo *li;
  DominatorTree *dt;
  RegionReconstructor *rr;
  ScopeTree *st;
};

}

//===------------------------ writeExpressions.h --------------------------===//
