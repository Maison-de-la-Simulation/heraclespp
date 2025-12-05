#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

import argparse
import configparser
import pathlib
import shutil
import subprocess
import tempfile

from check_convergence_advection_sinus import check_convergence_order


def run_convergence_test(sources: pathlib.Path):
    """Run convergence sinus test"""
    pdi_config_path = sources.joinpath(
        "test/convergence_advection_sinus/pdi_config.yaml"
    )
    binary_path = sources.joinpath("bin/heracles++")
    base_setup_config_path = sources.joinpath("inputs/advection_sinus.ini")

    try:
        test_dir_name = pathlib.Path(tempfile.mkdtemp())
        print(f"Creating temporary directory: {test_dir_name}")

        # Update the INI file and run commands
        def update_and_run(nx_glob: int, directory: pathlib.Path, prefix: str):
            setup_config_path = test_dir_name.joinpath(base_setup_config_path.name)
            config = configparser.ConfigParser(inline_comment_prefixes="#")
            config.read(base_setup_config_path)
            config["Grid"]["Nx0_glob"] = str(nx_glob)
            config["Output"]["directory"] = str(directory)
            config["Output"]["prefix"] = prefix
            with open(setup_config_path, mode="w", encoding="utf-8") as configfile:
                config.write(configfile)

            subprocess.run(
                [
                    "mpiexec",
                    "-n",
                    "2",
                    binary_path,
                    setup_config_path,
                    f"--pdi-config={pdi_config_path}",
                ],
                check=True,
            )

        filenames = []
        for i in range(5):
            nx_glob = 50 * 2**i
            prefix = f"convergence_test_{i}"
            update_and_run(nx_glob, test_dir_name, prefix)
            filenames.append(str(test_dir_name.joinpath(prefix + "_00000001.h5")))

        check_convergence_order(filenames=filenames)
    finally:
        print(f"Cleaning temporary directory: {test_dir_name}")
        shutil.rmtree(test_dir_name)


if __name__ == "__main__":

    def main():
        """main function"""
        parser = argparse.ArgumentParser(description="")
        parser.add_argument(
            "-S",
            default=pathlib.Path.cwd(),
            type=pathlib.Path,
            help="Path to the heracles++ sources",
        )
        args = parser.parse_args()

        run_convergence_test(args.S)

    main()
