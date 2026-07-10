"""
inspect_grib.py
----------------
Run this against ONE of your ECMWF S2S .grib files to print out its full
structure. ALWAYS enumerates every hypercube in the file (a plain
xr.open_dataset() call silently returns only ONE hypercube if a file packs
several incompatible ones -- this is very likely why t2m/d2m/CAPE/TCW
didn't show up earlier even though the filename implies they should be
there).

Usage:
    python inspect_grib.py "path/to/file.grib"
"""
import sys
import cfgrib


def inspect(path):
    print(f"\n{'='*70}\nFILE: {path}\n{'='*70}")

    datasets = cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""})
    print(f"\nFound {len(datasets)} hypercube(s) in this file.\n")

    for i, ds in enumerate(datasets):
        print(f"\n{'-'*70}\nHypercube {i}\n{'-'*70}")
        _describe(ds)


def _describe(ds):
    print("Data variables:")
    for name, da in ds.data_vars.items():
        print(f"  {name:12s} dims={da.dims}  shape={da.shape}  units={da.attrs.get('units', '?')}")

    print("\nCoordinates:")
    for name, coord in ds.coords.items():
        vals = coord.values
        size = getattr(vals, "size", 1)
        preview = vals if size <= 10 else f"{vals.flat[0]} ... {vals.flat[-1]} (n={size})"
        print(f"  {name:12s} {preview}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_grib.py <path_to_grib_file>")
        sys.exit(1)
    inspect(sys.argv[1])
