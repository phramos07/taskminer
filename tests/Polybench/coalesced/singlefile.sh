OPT="/home/brenocfg/Work/llvm-3.7/build-dawn/bin/opt"
CLANG="/home/brenocfg/Work/llvm-3.7/build-dawn/bin/clang"
CLANGF="/home/brenocfg/Work/llvm-3.7/build-dawn/bin/clang-format"
PGCC="/opt/pgi/linux86-64/2016/bin/pgcc"

LIB_DIR="/home/brenocfg/Work/Dawn/lge/lib"

ARRAY_LIB="$LIB_DIR/ArrayInference/libLLVMArrayInference.so"
PTR_LIB="$LIB_DIR/PtrRangeAnalysis/libLLVMPtrRangeAnalysis.so"
CANP_LIB="$LIB_DIR/CanParallelize/libCanParallelize.so"
DEPB_LIB="$LIB_DIR/DepBasedParallelLoopAnalysis/libParallelLoopAnalysis.so"
SCOPE_LIB="$LIB_DIR/ScopeTree/libLLVMScopeTree.so"
ALIAS_LIB="$LIB_DIR/AliasInstrumentation/libLLVMAliasInstrumentation.so"

INPUT="$1"
SIZE="$2"
COALESCE="$3"
OMP="$4"

echo "Running clang-format..."
$CLANGF -i -style="{BasedOnStyle: llvm, IndentWidth: 2}" $(pwd)/$INPUT

echo "Running scope-finder..."
bash scope.sh $(pwd)/$INPUT $SIZE

echo "Running Clang..."
$CLANG -g -c -S -emit-llvm $(pwd)/$INPUT $SIZE -o result.bc

echo "Running PtrRangeAnalysis..."
$OPT -load $PTR_LIB -load $ALIAS_LIB -load $DEPB_LIB -load $CANP_LIB -mem2reg \
-tbaa -scoped-noalias -basicaa -functionattrs -gvn -loop-rotate -instcombine \
-licm -ptr-ra -alias-instrumentation -region-alias-checks -can-parallelize \
result.bc

echo "Running ScopeTree/ParallelAnnotator..."
$OPT -load $SCOPE_LIB -load $ARRAY_LIB -annotateParallel -S result.bc \
-o result2.bc

echo "Running ArrayInference..."
$OPT -S -mem2reg -instnamer -loop-rotate -load $SCOPE_LIB -load $ARRAY_LIB \
-writeInFile -stats -Emit-GPU=true -Emit-Parallel=false -Emit-OMP=$OMP \
-Restrictifier=true -Memory-Coalescing=$COALESCE -Ptr-licm=true \
-Ptr-region=true result2.bc -o result3.bc

rm *.bc
rm *.log
find . -name "*.dot" -delete
