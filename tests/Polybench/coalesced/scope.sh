CLANG="/home/brenocfg/Work/llvm-3.7/test/bin/clang"
PLUGIN="/home/brenocfg/Work/llvm-3.7/test/lib/scope-finder.so"
OMP="-I/home/brenocfg/Work/Testing/UniBench/lib"
INPUT="$1"
SIZE="$2"

$CLANG $OMP $SIZE -Xclang -load -Xclang $PLUGIN -Xclang -add-plugin -Xclang -find-scope -g -O0 -fsyntax-only $INPUT
