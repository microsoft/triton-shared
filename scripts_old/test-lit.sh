cd /home/nhat/github/triton_shared/triton/python
LIT_TEST_DIR="build/$(ls build | grep -i cmake)/third_party/triton_shared/test"
if [ ! -d "${LIT_TEST_DIR}" ]; then
    echo "Coult not find '${LIT_TEST_DIR}'" ; exit -1
fi
lit -v "${LIT_TEST_DIR}"
