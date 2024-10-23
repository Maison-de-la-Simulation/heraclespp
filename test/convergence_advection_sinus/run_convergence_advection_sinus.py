#!/usr/bin/env python3

import configparser
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

        # Update the INI file and run commands
        def update_and_run(nx_glob, prefix):
            config = configparser.ConfigParser(inline_comment_prefixes='#')
            config.read(INI_FILENAME)
            config["Grid"]["Nx_glob"] = str(nx_glob)
            config["Output"]["directory"] = "."
            config["Output"]["prefix"] = prefix
            with open(INI_FILENAME, 'w') as configfile:
                config.write(configfile)

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
