# Installation script for mac M1
# Inspired by https://stackoverflow.com/questions/65745683/how-to-install-scipy-on-apple-silicon-arm-m1

# include pyenv commands here
pip3 install cython pybind11 # Required to compile numpy
pip3 install --no-binary :all: --no-use-pep517 numpy
pip3 install pythran # for scipy
brew install openblas gfortran # for scipy
export OPENBLAS=/opt/homebrew/opt/openblas/lib/
pip3 install --no-binary :all: --no-use-pep517 scipy
