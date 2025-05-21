#!/bin/bash

rm ./../../../optee/optee/out-br/build/optee_examples_ext-1.0/TorchZone -rf

rm ./../../../optee/optee/optee_examples/TorchZone/* -rf
# rm ./../../../optee/optee/optee_examples/TorchZone/host -rf
# rm ./../../../optee/optee/optee_examples/TorchZone/ta -rf
# rm ./../../../optee/optee/optee_examples/TorchZone/CMakeLists.txt
# rm ./../../../optee/optee/optee_examples/TorchZone/Makefile
cp ./host ./ta ./CMakeLists.txt ./Makefile ./../../../optee/optee/optee_examples/TorchZone/ -r

./installHeads.sh
rm ./../../../optee/optee/out-br/target/root/TExt -r
cp ./../PyTZone/TExt ./../../../optee/optee/out-br/target/root/ -r

rm ./../../../optee/optee/out-br/target/root/TExt/build_cmake -r
bash ./installHeads.sh