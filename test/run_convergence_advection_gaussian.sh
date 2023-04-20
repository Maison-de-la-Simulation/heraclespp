#! /bin/bash

#set -ex

#TEST_DIR_NAME=convergence_advection_sinusoide
#PDI_YML_FILENAME=convergence_advection_sinusoide.yml
#BINARY_NAME=nova++
#INI_FILENAME=advection_sinus.ini

#mkdir $TEST_DIR_NAME && cd $TEST_DIR_NAME
#cp ../inputs/$INI_FILENAME .
#cp ../bin/$BINARY_NAME .
#cp ../test/$PDI_YML_FILENAME .
#sed -i 's/Nx_glob = .*/Nx_glob = 50/g' ./$INI_FILENAME
#mpiexec -n 2 ./$BINARY_NAME ./$INI_FILENAME ./$PDI_YML_FILENAME
#sed -i 's/Nx_glob = .*/Nx_glob = 100/g' ./$INI_FILENAME
#mpiexec -n 2 ./$BINARY_NAME ./$INI_FILENAME ./$PDI_YML_FILENAME
#sed -i 's/Nx_glob = .*/Nx_glob = 200/g' ./$INI_FILENAME
#mpiexec -n 2 ./$BINARY_NAME ./$INI_FILENAME ./$PDI_YML_FILENAME
#sed -i 's/Nx_glob = .*/Nx_glob = 400/g' ./$INI_FILENAME
#mpiexec -n 2 ./$BINARY_NAME ./$INI_FILENAME ./$PDI_YML_FILENAME
#sed -i 's/Nx_glob = .*/Nx_glob = 800/g' ./$INI_FILENAME
#mpiexec -n 2 ./$BINARY_NAME ./$INI_FILENAME ./$PDI_YML_FILENAME
#python3 ../test/check_convergence_advection_sinusoide.py

set -ex

TEST_DIR_NAME=convergence_advection_gaussian
PDI_YML_FILENAME=convergence_advection_gaussian.yml
BINARY_NAME=nova++
INI_FILENAME=advection_gaussian.ini

mkdir $TEST_DIR_NAME && cd $TEST_DIR_NAME
cp ../inputs/$INI_FILENAME .
cp ../bin/$BINARY_NAME .
cp ../test/$PDI_YML_FILENAME .
sed -i.bak 's/Nx_glob = .*/Nx_glob = 50/g' ./$INI_FILENAME
mpiexec -n 2 ./$BINARY_NAME ./$INI_FILENAME ./$PDI_YML_FILENAME
sed -i.bak 's/Nx_glob = .*/Nx_glob = 100/g' ./$INI_FILENAME
mpiexec -n 2 ./$BINARY_NAME ./$INI_FILENAME ./$PDI_YML_FILENAME
sed -i.bak 's/Nx_glob = .*/Nx_glob = 200/g' ./$INI_FILENAME
mpiexec -n 2 ./$BINARY_NAME ./$INI_FILENAME ./$PDI_YML_FILENAME
sed -i.bak 's/Nx_glob = .*/Nx_glob = 400/g' ./$INI_FILENAME
mpiexec -n 2 ./$BINARY_NAME ./$INI_FILENAME ./$PDI_YML_FILENAME
sed -i.bak 's/Nx_glob = .*/Nx_glob = 800/g' ./$INI_FILENAME
mpiexec -n 2 ./$BINARY_NAME ./$INI_FILENAME ./$PDI_YML_FILENAME
python3 ../test/check_convergence_advection_gaussian.py