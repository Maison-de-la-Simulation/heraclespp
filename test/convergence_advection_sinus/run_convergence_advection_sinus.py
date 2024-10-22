#!/usr/bin/env python3

import os
import shutil
import subprocess
import tempfile

def main():
    PDI_YAML_FILENAME = "convergence_advection_sinus.yaml"
    BINARY_NAME = "nova++"
    INI_FILENAME = "advection_sinus.ini"

    try:
        TEST_DIR_NAME = tempfile.mkdtemp()
        print(f"Creating temporary directory: {TEST_DIR_NAME}")

        # Copy necessary files
        shutil.copy(f"./inputs/{INI_FILENAME}", TEST_DIR_NAME)
        shutil.copy(f"./bin/{BINARY_NAME}", TEST_DIR_NAME)
        shutil.copy(f"./test/convergence_advection_sinus/{PDI_YAML_FILENAME}", TEST_DIR_NAME)
        shutil.copy(f"./test/convergence_advection_sinus/check_convergence_advection_sinus.py", TEST_DIR_NAME)

        os.chdir(TEST_DIR_NAME)
        subprocess.run(["sed", "-i.bak", "s/directory = .*/directory = ./g", INI_FILENAME], check=True)

        # Update the INI file and run commands
        def update_and_run(nx_glob, prefix):
            subprocess.run(["sed", "-i.bak", f"s/Nx_glob = .*/Nx_glob = {nx_glob}/g", INI_FILENAME], check=True)
            subprocess.run(["sed", "-i.bak", f"s/prefix = .*/prefix = {prefix}/g", INI_FILENAME], check=True)
            subprocess.run(["mpiexec", "-n", "2", BINARY_NAME, INI_FILENAME, f"--pdi-config=./{PDI_YAML_FILENAME}"], check=True)

        for i in range(5):
            nx_glob = 50*2**i
            update_and_run(nx_glob, f"convergence_test_advection_sinus_{i}")

        # Run the Python script to check convergence
        subprocess.run(["python3", "./check_convergence_advection_sinus.py"], check=True)

    finally:
        print(f"Cleaning temporary directory: {TEST_DIR_NAME}")
        shutil.rmtree(TEST_DIR_NAME)

if __name__ == "__main__":
    main()
