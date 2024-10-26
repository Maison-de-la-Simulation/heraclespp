#!/usr/bin/env python3

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile
import yaml

def build_suite(setups):
    try:
        # Create a temporary directory
        directory = pathlib.Path(tempfile.mkdtemp())
        print(f"Creating temporary directory: {directory}")
        build_directory = directory.joinpath("build")

        for setup in setups:
            cmake_options = setup["cmake_options"]
            subprocess.run(["cmake",
                        f"-DBUILD_TESTING=OFF",
                        f"-DNovapp_EOS={cmake_options['eos']}",
                        f"-DNovapp_GEOM={cmake_options['geom']}",
                        f"-DNovapp_GRAVITY={cmake_options['gravity']}",
                        f"-DNovapp_GTest_DEPENDENCY_POLICY=INSTALLED",
                        f"-DNovapp_inih_DEPENDENCY_POLICY=INSTALLED",
                        f"-DNovapp_Kokkos_DEPENDENCY_POLICY=INSTALLED",
                        f"-DNovapp_NDIM={cmake_options['ndim']}",
                        f"-DNovapp_SETUP={setup['name']}",
                        "-B", build_directory])
            subprocess.run(["cmake", "--build", build_directory], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}")
        sys.exit(1)
    finally:
        print(f"Cleaning temporary directory: {directory}")
        shutil.rmtree(directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build suite")
    parser.add_argument("filename",
                        type=pathlib.Path,
                        help="Input YAML filename")
    args = parser.parse_args()

    with open(args.filename, "r") as yaml_file:
        setups = yaml.safe_load(yaml_file)

    build_suite(setups)
