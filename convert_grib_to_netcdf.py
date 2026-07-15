"""
convert_grib_to_netcdf.py
----------------------------
Converts a GRIB file into one NetCDF file PER VARIABLE, chunked to match
how grib_dataset.py actually reads data: one (time, step, member[, level])
slice at a time, but always the FULL lat/lon grid.

Chunking rationale:
  - number/time/step/isobaricInhPa: chunk size 1 -- we only ever read a
    single index along these dims per sample, so small chunks mean each
    read touches the minimum data on disk.
  - latitude/longitude: chunk size = full dimension length -- we always
    want the whole spatial field, so there's no benefit to sub-chunking it
    (and sub-chunking it would just add overhead for no gain).

Enumerates every hypercube via cfgrib.open_datasets() rather than a plain
xr.open_dataset() call, since a single GRIB file can pack several
incompatible hypercubes and a plain open call silently returns only one
of them (this bit us earlier with the sfc_d2m_t2m_CAPE_TCW file).

Usage:
    python convert_grib_to_netcdf.py --input path/to/file.grib --output_dir ./netcdf_out

    # Convert only specific variables (skip the rest):
    python convert_grib_to_netcdf.py --input path/to/file.grib --output_dir ./netcdf_out \\
        --variables mx2t6 mn2t6
"""
import argparse
import os
import time

import cfgrib
import xarray as xr


CHUNK1_DIMS = {"number", "time", "step", "isobaricInhPa"}


def build_chunk_plan(da):
    """One chunk-size-1 slice per (number/time/step/level) dim, full size
    for every other dim (lat/lon, or whatever else shows up)."""
    chunks = {}
    for dim in da.dims:
        if dim in CHUNK1_DIMS:
            chunks[dim] = 1
        else:
            chunks[dim] = da.sizes[dim]
    return chunks


def convert(input_path, output_dir, only_variables=None):
    os.makedirs(output_dir, exist_ok=True)

    print(f">> Opening {input_path} (enumerating all hypercubes -- this can take a while for large files)")
    t0 = time.time()
    hypercubes = cfgrib.open_datasets(input_path, backend_kwargs={"indexpath": ""})
    print(f">> Found {len(hypercubes)} hypercube(s) in {time.time() - t0:.1f}s")

    converted = []
    for i, ds in enumerate(hypercubes):
        for var_name, da in ds.data_vars.items():
            if only_variables and var_name not in only_variables:
                continue

            print(f"\n>> [{var_name}] hypercube {i}, dims={da.dims}, shape={da.shape}")
            chunk_plan = build_chunk_plan(da)
            print(f"   chunk plan: {chunk_plan}")

            out_path = os.path.join(output_dir, f"{var_name}.nc")
            encoding = {
                var_name: {
                    "zlib": True,
                    "complevel": 4,
                    "chunksizes": tuple(chunk_plan[dim] for dim in da.dims),
                }
            }

            t1 = time.time()
            out_ds = da.to_dataset(name=var_name)
            out_ds.to_netcdf(out_path, encoding=encoding)
            elapsed = time.time() - t1
            size_mb = os.path.getsize(out_path) / 1e6
            print(f"   -> saved to {out_path} ({size_mb:.1f} MB) in {elapsed:.1f}s")
            converted.append(var_name)

    print(f"\n>> Done. Converted {len(converted)} variable(s): {converted}")
    if not converted:
        print(">> WARNING: nothing was converted. Check --variables against the "
              "actual shortNames in the file (run inspect_grib.py or "
              "list_grib_messages.py first if unsure).")


def main():
    parser = argparse.ArgumentParser(description="Convert a GRIB file to per-variable NetCDF, chunked for fast single-sample reads")
    parser.add_argument("--input", type=str, required=True, help="Path to the .grib file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write one .nc file per variable")
    parser.add_argument("--variables", type=str, nargs="*", default=None,
                         help="Optional: only convert these shortNames (default: convert everything found)")
    args = parser.parse_args()

    convert(args.input, args.output_dir, only_variables=set(args.variables) if args.variables else None)


if __name__ == "__main__":
    main()
