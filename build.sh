export TRITON_PLUGIN_DIRS="/home/nhat/github/triton_shared"
cd /home/nhat/github/triton_shared/triton/python
# python3 -m pip install --upgrade pip
# python3 -m pip install cmake==3.24
# python3 -m pip install ninja
# python3 -m pip uninstall -y triton
python3 setup.py build
# python3 -m pip install --no-build-isolation -vvv '.[tests]'
