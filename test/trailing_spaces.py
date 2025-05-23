#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

"""
Script to find trailing whitespaces.
"""

import argparse
import re
import sys


def find_trailing_whitespaces(files):
    """
    Find all lines with trailing whitespaces in C++ and CMake files within the given directory.

    Parameters
    ----------
    directory : str
        The directory to search for files.

    Returns
    -------
    list
        A list of tuples containing the filename, line number, and line content with trailing
        whitespaces.
    """
    files_with_trailing_whitespaces = []

    # Check each file for trailing whitespaces
    for filename in files:
        with open(filename, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if re.search(r"[ \t]+$", line):
                    files_with_trailing_whitespaces.append((filename, line_number, line.rstrip()))

    return files_with_trailing_whitespaces


def remove_trailing_whitespaces(files_with_trailing_whitespaces):
    """
    Remove trailing whitespaces from the specified files.

    Parameters
    ----------
    files_with_trailing_whitespaces : list
        A list of tuples containing the filename, line number, and line content with trailing
        whitespaces.
    """
    for filename, _, _ in files_with_trailing_whitespaces:
        with open(filename, "r", encoding="utf-8") as file:
            lines = file.readlines()

        with open(filename, "w", encoding="utf-8") as file:
            for line in lines:
                file.write(re.sub(r"[ \t]+$", "", line))

        # print(f"Removed trailing whitespaces from: {filename}")


def main():
    """
    The main function to parse arguments and find/remove trailing whitespaces.
    """
    parser = argparse.ArgumentParser(
        description="Find and optionally remove trailing whitespaces in given files."
    )
    parser.add_argument("files", type=str, nargs="+", help="Files to analyze.")
    parser.add_argument("-i", action="store_true", help="Remove trailing whitespaces.")
    parser.add_argument("--Werror", action="store_true", help="If set, treat warnings as errors")
    args = parser.parse_args()

    files_with_trailing_whitespaces = find_trailing_whitespaces(args.files)

    if args.i:
        remove_trailing_whitespaces(files_with_trailing_whitespaces)
    else:
        if files_with_trailing_whitespaces:
            for file, line_number, line in files_with_trailing_whitespaces:
                print(f"{file}:{line_number}:\n{line}")

            if args.Werror:
                sys.exit(1)


if __name__ == "__main__":
    main()
