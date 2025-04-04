#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

"""
This script automates the process of adding reuse annotations to files.
It navigates two directories up from the script's location and runs the
'reuse annotate' command with predefined options.
"""

import argparse
import subprocess


def main():
    """
    Main function to run the reuse annotate command with specified options.
    """
    # Define the copyright and license text
    copyright_text = "The HERACLES++ development team, see COPYRIGHT.md file"
    license_text = "MIT"

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Annotate files using reuse tool.")
    parser.add_argument("files", nargs="+", help="Files to annotate")
    args = parser.parse_args()

    # Construct the command
    command = [
        "reuse",
        "annotate",
        "--copyright",
        copyright_text,
        "--license",
        license_text,
        "--merge-copyrights",
        "--recursive",
        "--skip-unrecognised",
        *args.files,
    ]

    # Execute the command
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
