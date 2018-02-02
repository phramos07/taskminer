THIS=$(pwd)

FILES_ACC=$(find . -path "*acc*.c" ! -name "*_*.c")
FILES_OMP=$(find . -path "*omp*.c" ! -name "*_*.c")

for f in $FILES_ACC;
do
    NAME=$f
    EXT="${NAME##*.}"
    NAME="${NAME%.*}"
    echo "FILE = $f"
    bash singlefile.sh $f -DMINI_DATASET true 0
    mv $NAME"_AI."$EXT $NAME"_MINI."$EXT
    bash singlefile.sh $f -DSMALL_DATASET true 0
    mv $NAME"_AI."$EXT $NAME"_SMALL."$EXT
    bash singlefile.sh $f -DMEDIUM_DATASET true 0
    mv $NAME"_AI."$EXT $NAME"_MEDIUM."$EXT
    bash singlefile.sh $f -DLARGE_DATASET true 0
    mv $NAME"_AI."$EXT $NAME"_LARGE."$EXT
    bash singlefile.sh $f -DEXTRALARGE_DATASET true 0
    mv $NAME"_AI."$EXT $NAME"_EXTRALARGE."$EXT
done

for f in $FILES_OMP;
do
    NAME=$f
    EXT="${NAME##*.}"
    NAME="${NAME%.*}"
    bash singlefile.sh $f -DMINI_DATASET true 2
    mv $NAME"_AI."$EXT $NAME"_MINI."$EXT
    bash singlefile.sh $f -DSMALL_DATASET true 2
    mv $NAME"_AI."$EXT $NAME"_SMALL."$EXT
    bash singlefile.sh $f -DMEDIUM_DATASET true 2
    mv $NAME"_AI."$EXT $NAME"_MEDIUM."$EXT
    bash singlefile.sh $f -DLARGE_DATASET true 2
    mv $NAME"_AI."$EXT $NAME"_LARGE."$EXT
    bash singlefile.sh $f -DEXTRALARGE_DATASET true 2
    mv $NAME"_AI."$EXT $NAME"_EXTRALARGE."$EXT
done

echo $FILES
