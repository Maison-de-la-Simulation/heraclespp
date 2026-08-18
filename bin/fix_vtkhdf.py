#!/usr/bin/env python3

import argparse
import glob

import h5py


def convert_type_attribute(filename, dry_run=False):
    with h5py.File(filename, "r" if dry_run else "r+") as f:
        vtkhdf = f["VTKHDF"]

        if "Type" not in vtkhdf.attrs:
            print(f"  Skipping {filename}: Type attribute not found")
            return

        attribute = vtkhdf.attrs["Type"]

        # Already a string: nothing to do.
        if isinstance(attribute, str):
            print(f"  Skipping {filename}: Type is already a string ('{attribute}')")
            return

        # Handle bytes/string arrays as well as uint8 arrays.
        if attribute.dtype.kind == "u":
            value = bytes(attribute).decode("utf-8")
        elif attribute.dtype.kind == "S":
            value = b"".join(attribute.flat).decode("utf-8")
        else:
            print(f"  Skipping {filename}: Type has unexpected dtype {attribute.dtype}")
            return

        if dry_run:
            print(
                f"  Would convert {filename}: "
                f"Type {attribute.dtype} -> string '{value}'"
            )
            return

        del vtkhdf.attrs["Type"]
        vtkhdf.attrs.create(
            "Type",
            value,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

        print(f"  Converted {filename}: Type -> '{value}'")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert the VTKHDF Type attribute from a character array "
            "to an HDF5 string."
        )
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="HDF5 files or glob patterns to convert",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )

    args = parser.parse_args()

    files = []
    for pattern in args.files:
        matches = glob.glob(pattern, recursive=True)

        if matches:
            files.extend(matches)
        else:
            print(f"Warning: no files matched '{pattern}'")

    # Remove duplicates while preserving order.
    files = list(dict.fromkeys(files))

    if not files:
        print("No files to process.")
        return

    if args.dry_run:
        print("Dry run: no files will be modified.")

    for filename in files:
        convert_type_attribute(filename, args.dry_run)


if __name__ == "__main__":
    main()
