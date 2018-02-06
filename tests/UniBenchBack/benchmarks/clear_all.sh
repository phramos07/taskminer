#!/bin/bash
THIS=`pwd`

#cd /home/gleison/lge/llvm-3.7-src/build-debug/bin 

#rm out.txt

#cd -

#rm out.txt benchmarksStatistics.csv

# List of benchmarks available to use our techinic
list=(
    #OMPSpec Benchmarks
    'OMPSpec/target-3c/src/target-3c'
    'OMPSpec/target-data-1c/src/target-data-1c'
    'OMPSpec/target-data-2c/src/target-data-2c'
    'OMPSpec/target-data-6c/src/target-data-6c'
    #Parboil Benchmarks
    'Parboil/common/parboil'
    'Parboil/mri-q/src/main'
    'Parboil/mri-q/src/file'
    'Parboil/sgemm/src/io'
    'Parboil/sgemm/src/main'
    'Parboil/sgemm/src/sgemm_kernel'
    'Parboil/spmv/src/convert_dataset'
    'Parboil/spmv/src/file'
    'Parboil/spmv/src/main'
    'Parboil/spmv/src/mmio'
    'Parboil/stencil/src/file'
    'Parboil/stencil/src/kernels'
    'Parboil/stencil/src/main'
    #Polybench Benchmarks
    'Polybench/2DCONV/src/2DConvolution'
    'Polybench/2MM/src/2mm'
    'Polybench/3DCONV/src/3DConvolution'
    'Polybench/3MM/src/3mm'
    'Polybench/ATAX/src/atax'
    'Polybench/BICG/src/bicg'
    'Polybench/CORR/src/correlation'
    'Polybench/COVAR/src/covariance'
    'Polybench/FDTD-2D/src/fdtd2d'
    'Polybench/GEMM/src/gemm'
    'Polybench/GESUMMV/src/gesummv'
    'Polybench/GRAMSCHM/src/gramschmidt'
    'Polybench/MVT/src/mvt'
    'Polybench/SYR2K/src/syr2k'
    'Polybench/SYRK/src/syrk'
    'Polybench/SYRK_M/src/syrk_m'
    #Rodinia 
    'Rodinia/backprop/src/backprop'
    'Rodinia/backprop/src/backprop_kernel'
    'Rodinia/backprop/src/facetrain'
    'Rodinia/backprop/src/imagenet'
    'Rodinia/bfs/src/bfs'
    'Rodinia/b+tree/src/main'
    'Rodinia/b+tree/src/kernel/kernel_cpu'
    'Rodinia/b+tree/src/kernel/kernel_cpu_2'   
    # C++ code: 'Rodinia/hotspot/src/hotspot_openmp'
    'Rodinia/lud/src/lud'
    'Rodinia/lud/src/lud_omp'
    'Rodinia/lud/src/tools/gen_input'
    # C++ code: 'Rodinia/nw/src/needle'
    'Rodinia/srad/src/define'
    'Rodinia/srad/src/graphics'
    'Rodinia/srad/src/main'
    'Rodinia/srad/src/resize'
    'Rodinia/srad/src/timer'
    #MG BENCH
    'mgBench/collinear-list/src/collinear-list_gpu'
    'mgBench/cholesky/src/cholesky_gpu'
    'mgBench/floyd/src/floyd_gpu'
    'mgBench/k-nearest/src/k-nearest_gpu'
    'mgBench/lu-decomposition/src/lu-decomposition_gpu'
    'mgBench/mat-mul/src/mat-mul_gpu'
    'mgBench/mat-sum/src/mat-sum_gpu'
    'mgBench/other-nearest/src/other-nearest_gpu'
    'mgBench/search-vector/src/search-vector_gpu'
    'mgBench/str-matching/src/str-matching_gpu'
    'mgBench/vector-product/src/vector-product_gpu'
)

testList=(
    #OMPSpec Benchmarks
    'OMPSpec/target-3c/src/target-3c'
    'OMPSpec/target-data-1c/src/target-data-1c'
    'OMPSpec/target-data-2c/src/target-data-2c'
    'OMPSpec/target-data-6c/src/target-data-6c'
    #MG BENCH
    'mgBench/collinear-list/src/collinear-list_gpu'
    'mgBench/cholesky/src/cholesky_gpu'
    'mgBench/floyd/src/floyd_gpu'
    'mgBench/k-nearest/src/k-nearest_gpu'
    'mgBench/lu-decomposition/src/lu-decomposition_gpu'
    'mgBench/mat-mul/src/mat-mul_gpu'
    'mgBench/mat-sum/src/mat-sum_gpu'
    'mgBench/search-vector/src/search-vector_gpu'
    'mgBench/str-matching/src/str-matching_gpu'
    'mgBench/vector-product/src/vector-product_gpu'
    #Polybench Benchmarks
    'Polybench/2DCONV/src/2DConvolution'
    'Polybench/2MM/src/2mm'
    'Polybench/3DCONV/src/3DConvolution'
    'Polybench/3MM/src/3mm'
    'Polybench/ATAX/src/atax'
    'Polybench/BICG/src/bicg'
    'Polybench/CORR/src/correlation'
    'Polybench/COVAR/src/covariance'
    'Polybench/FDTD-2D/src/fdtd2d'
    'Polybench/GEMM/src/gemm'
    'Polybench/GESUMMV/src/gesummv'
    'Polybench/GRAMSCHM/src/gramschmidt'
    'Polybench/MVT/src/mvt'
    'Polybench/SYR2K/src/syr2k'
    'Polybench/SYRK/src/syrk'
    'Polybench/SYRK_M/src/syrk_m'
)

inputList=(
    #OMPSpec Benchmarks
    ''
    ''
    ''
    ''
    '101'
    '101'
    '101'
    '101'
    '101'
    '101'
    '101'
    '101'
    '101'
    '101'
    '101'
    ''
    ''
    ''
    ''
    ''
    ''
    ''
    ''
    ''
    ''
    ''
    ''
    ''
    ''
    ''
)

echo 'Running Array Inference'
count=0
while [ "x${list[count]}" != "x" ]
do
  echo "Trying to  $(count) : ${list[count]}.c"
  ./removePrag.sh "${list[count]}.c"

  count=$(( $count + 1 ))
done  

#./STATISTICS out.txt

#echo 'Compiling and Running Array Inference Parallel Programs'
#count=0
#while [ "x${testList[count]}" != "x" ]
#do
  #ipmacc ${testList[count]}'_AI.c' -o TEST
  #ipmacc ${testList[count]}'_AI.cc' -o TEST

  #(./TEST inputList[count] &>> errorsFile.txt) &>> errorsFile2.txt

#  count=$(( $count + 1 ))
#done

#echo 'Running Parboil'

#ipmacc Parboil/mri-q/src/main_AI.c -o TEST
#./TEST Parboil/mri-q/src/input/64_64_64_dataset.bin

#ipmacc Parboil/sgemm/src/main_AI.cc -o TEST
#./TEST Parboil/sgemm/src/input/matrix1.txt

