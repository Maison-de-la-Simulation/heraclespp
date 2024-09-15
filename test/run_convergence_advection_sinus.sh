#! /bin/bash

TEST_DIR_NAME=convergence_advection_sinus
PDI_YML_FILENAME=convergence_advection_sinus.yml
BINARY_NAME=nova++
INI_FILENAME=advection_sinus.ini

TEST_DIR_NAME=$(mktemp -d)
if [ $? -ne 0 ]; then
    echo "$0: Can't create temporary directory, exiting."
    exit 1
fi

trap 'rm -rf -- "$TEST_DIR_NAME"' EXIT
set -ex

cp ./inputs/$INI_FILENAME $TEST_DIR_NAME
cp ./bin/$BINARY_NAME $TEST_DIR_NAME
cp ./test/$PDI_YML_FILENAME $TEST_DIR_NAME
cp ./test/check_convergence_advection_sinus.py $TEST_DIR_NAME

cd $TEST_DIR_NAME
sed -i.bak 's/directory = .*/directory = ./g' ./$INI_FILENAME

sed -i.bak 's/Nx_glob = .*/Nx_glob = 50/g' ./$INI_FILENAME
sed -i.bak 's/prefix = .*/prefix = convergence_test_advection_sinus_1/g' ./$INI_FILENAME
mpiexec -n 2 ./$BINARY_NAME ./$INI_FILENAME --pdi-config=./$PDI_YML_FILENAME
sed -i.bak 's/Nx_glob = .*/Nx_glob = 100/g' ./$INI_FILENAME
sed -i.bak 's/prefix = .*/prefix = convergence_test_advection_sinus_2/g' ./$INI_FILENAME
mpiexec -n 2 ./$BINARY_NAME ./$INI_FILENAME --pdi-config=./$PDI_YML_FILENAME
sed -i.bak 's/Nx_glob = .*/Nx_glob = 200/g' ./$INI_FILENAME
sed -i.bak 's/prefix = .*/prefix = convergence_test_advection_sinus_3/g' ./$INI_FILENAME
mpiexec -n 2 ./$BINARY_NAME ./$INI_FILENAME --pdi-config=./$PDI_YML_FILENAME
sed -i.bak 's/Nx_glob = .*/Nx_glob = 400/g' ./$INI_FILENAME
sed -i.bak 's/prefix = .*/prefix = convergence_test_advection_sinus_4/g' ./$INI_FILENAME
mpiexec -n 2 ./$BINARY_NAME ./$INI_FILENAME --pdi-config=./$PDI_YML_FILENAME
sed -i.bak 's/Nx_glob = .*/Nx_glob = 800/g' ./$INI_FILENAME
sed -i.bak 's/prefix = .*/prefix = convergence_test_advection_sinus_5/g' ./$INI_FILENAME
mpiexec -n 2 ./$BINARY_NAME ./$INI_FILENAME --pdi-config=./$PDI_YML_FILENAME

python3 ./check_convergence_advection_sinus.py

rm -rf -- "$TEST_DIR_NAME"

trap - EXIT