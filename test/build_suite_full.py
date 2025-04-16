#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile
import typing

import yaml


def build_suite(sources: pathlib.Path, setups: typing.List[typing.Dict]):
    """
    Function that builds the different configurations found in setups.
    It assumes the dependencies are already installed.
    """
    try:
        # Create a temporary directory
        directory = pathlib.Path(tempfile.mkdtemp())
        print(f"Creating temporary directory: {directory}")
        build_directory = directory.joinpath("build")

        for setup in setups:
            cmake_options = setup["cmake_options"]
            subprocess.run(
                [
                    "cmake",
                    "-DBUILD_TESTING=OFF",
                    f"-DNovapp_EOS={cmake_options['eos']}",
                    f"-DNovapp_GEOM={cmake_options['geom']}",
                    f"-DNovapp_GRAVITY={cmake_options['gravity']}",
                    "-DNovapp_GTest_DEPENDENCY_POLICY=INSTALLED",
                    "-DNovapp_inih_DEPENDENCY_POLICY=INSTALLED",
                    "-DNovapp_Kokkos_DEPENDENCY_POLICY=INSTALLED",
                    f"-DNovapp_NDIM={cmake_options['ndim']}",
                    f"-DNovapp_SETUP={setup['name']}",
                    "-B",
                    build_directory,
                    "-S",
                    sources,
                ],
                check=True,
            )
            subprocess.run(["cmake", "--build", build_directory], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}")
        sys.exit(1)
    finally:
        print(f"Cleaning temporary directory: {directory}")
        shutil.rmtree(directory)


if __name__ == "__main__":

    def main():
        """main function"""
        parser = argparse.ArgumentParser(description="Build suite")
        parser.add_argument(
            "filename", type=pathlib.Path, help="Path to input YAML configuration filename"
        )
        parser.add_argument(
            "-S", default=pathlib.Path.cwd(), type=pathlib.Path, help="Path to the nova++ sources"
        )
        args = parser.parse_args()

        with open(args.filename, encoding="utf-8") as yaml_file:
            setups = yaml.safe_load(yaml_file)

        build_suite(args.S, setups)

    main()
