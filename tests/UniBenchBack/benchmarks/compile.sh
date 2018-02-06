#!/bin/bash
THIS=`pwd`
TOPLEVEL=${THIS}/../../../
#===----------------------------------------------------------------------===//
#             Change to use in your Benchmarks paste
#===----------------------------------------------------------------------===//

BENCH=${THIS}/

#===----------------------------------------------------------------------===//
#
#===----------------------------------------------------------------------===//
COMP=${TOPLEVEL}/llvm-3.7-src/build-debug/bin/
LIBR=${TOPLEVEL}/dawn-3.7/build-debug/ArrayInference/
LIBRNIKE=${TOPLEVEL}/nike-3.7/build-debug/

if [ $1 == "options" ]; then
  echo ""
  echo "  RN => to run the RecoverNames pass."
  echo ""
  echo "  WE => to run the WriteExpressions pass."
  echo ""
  echo "  WF => to run the WriteInFile pass."
  echo ""
  echo "  rmode => to create a binary code readable."
  echo ""
  echo "  B => to build the LLVM again."
  echo ""
  echo "  BA => to build the ArrayInference again."
  echo ""
  echo "  BN => to build the project NIKE again."
  echo ""
fi

if [ $1 == "RN" ]; then

rm ${BENCH}/$2.bc

cd ${COMP}

./clang -g -c -O0 -emit-llvm ${BENCH}/$2.c -o ${BENCH}/$2.bc

./opt -load ${LIBRNIKE}/PtrRangeAnalysis/libLLVMPtrRangeAnalysis.so \
-load ${LIBRNIKE}/dbg/libPtrRangeAnalysisDBG.so \
-load ${LIBR}/libLLVMArrayInference.so -RecoverNames ${BENCH}/$2.bc

#./bin/opt -load lib/LLVMArrayInference.so -array-size-inference ../llvm/mgBench/$2.bc -print-after-all

cd -

fi

if [ $1 == "WE" ]; then

rm ${BENCH}/$2.bc

cd ${COMP}

./clang -g -c -O0 -emit-llvm ${BENCH}/$2.c -o ${BENCH}/$2.bc

#./opt -O3 --debug-pass=Structure -load ${LIBRNIKE}/PtrRangeAnalysis/libLLVMPtrRangeAnalysis.so \
./opt -mem2reg \
-load ${LIBRNIKE}/PtrRangeAnalysis/libLLVMPtrRangeAnalysis.so \
-load ${LIBRNIKE}/dbg/libPtrRangeAnalysisDBG.so \
-load ${LIBR}/libLLVMArrayInference.so -writeExpressions ${BENCH}/$2.bc
#-load ${LIBR}/libLLVMArrayInference.so -writeExpressions ${BENCH}/$2.bc

#./opt -debug-pass=Structure -load libLLVMAnalysis.a  -load ../lib/LLVMArrayInference.so -writeExpressions ../../mgBench/$2.bc

#./bin/opt -load lib/LLVMArrayInference.so -array-size-inference ../llvm/mgBench/$2.bc -print-after-all

cd -

fi

if [ $1 == "WF" ]; then

rm ${BENCH}/$2.bc

cd ${COMP}

echo '/'$2

./clang -g -c -O0 -emit-llvm ${BENCH}/$2.c -o ${BENCH}/$2.bc
./clang++ -g -c -O0 -emit-llvm ${BENCH}/$2.cc -o ${BENCH}/$2.bc

./opt -mem2reg -instnamer -loop-rotate -load ${LIBR}/libLLVMArrayInference.so \
  -stats -writeInFile -Emit-GPU=false -Emit-Parallel=false ${BENCH}/$2.bc 2>> out.txt

cp out.txt ${THIS}/.

cd -

fi

if [ $1 == "rmode" ]; then

cd ${COMP}

rm ${BENCH}/$2.ll
./clang -g -S -O0 -c -emit-llvm ${BENCH}/$2.c -o ${BENCH}/$2.ll

./opt -S -mem2reg ${BENCH}/$2.ll -o ${BENCH}/$2'-out'.ll

vi ${BENCH}/$2'-out'.ll

cd -

fi

if [ $1 == "B" ]; then

cd ${COMP}/../

make -j8

cd -

fi

if [ $1 == "BA" ]; then

cd ${LIBR}/../

make -j8

cd -

fi

if [ $1 == "BN" ]; then

  cd ${LIBRNIKE}/

make -j8

cd -

fi

