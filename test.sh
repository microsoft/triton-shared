#!/usr/bin/bash
# cd /home/nhat/github/triton_old/third_party/triton_shared/test/Conversion/TritonToLinalg
# for f in $(ls); do
#     /home/nhat/github/triton_old/python/build/cmake.linux-x86_64-cpython-3.9/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize $f -o=/home/nhat/github/triton_old/third_party/triton_shared/test_outs/$f
# done

dir=/home/nhat/github/triton_shared/test/Conversion/TritonToLinalg
cd $dir
for f in $(ls); do
    /home/nhat/github/triton_shared/triton/python/build/cmake.linux-x86_64-cpython-3.9/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt --split-input-file --triton-to-structured --canonicalize --triton-arith-to-linalg --structured-to-memref $f -o=/home/nhat/github/triton_shared/_in/${f} &> /dev/null
    retVal=$?
    if [ $retVal -ne 0 ]; then
        echo $dir/$f
        echo "-----"
    fi
done
