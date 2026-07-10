"""
list_grib_messages.py
----------------------
Bypasses cfgrib's higher-level dataset construction entirely and reads the
RAW message inventory directly via eccodes. Use this when cfgrib's
open_datasets() shows fewer variables than a filename implies -- it tells
you definitively whether the missing variables are:
  (a) genuinely absent from the file, or
  (b) present as raw GRIB messages but silently dropped by cfgrib because
      it couldn't merge them into a coherent hypercube (e.g. unsupported
      paramId, mismatched grid, or incompatible dimensions).

Usage:
    python list_grib_messages.py "path/to/file.grib"
"""
import sys
from collections import Counter
import eccodes


def list_messages(path):
    print(f"\n{'='*70}\nRAW MESSAGE INVENTORY: {path}\n{'='*70}")

    short_names = Counter()
    details = []

    with open(path, "rb") as f:
        msg_count = 0
        while True:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                break
            msg_count += 1
            try:
                short_name = eccodes.codes_get(gid, "shortName")
                param_id = eccodes.codes_get(gid, "paramId")
                type_of_level = eccodes.codes_get(gid, "typeOfLevel")
                level = eccodes.codes_get(gid, "level")
                step_type = eccodes.codes_get(gid, "stepType")
                data_date = eccodes.codes_get(gid, "dataDate")
                step = eccodes.codes_get(gid, "step")
                short_names[short_name] += 1
                if short_names[short_name] <= 1:  # print first occurrence of each
                    details.append(
                        f"  shortName={short_name:12s} paramId={param_id:6d} "
                        f"typeOfLevel={type_of_level:15s} level={level:5d} "
                        f"stepType={step_type:8s} dataDate={data_date} step={step}"
                    )
            except Exception as e:
                details.append(f"  [message {msg_count}] could not read keys: {e}")
            finally:
                eccodes.codes_release(gid)

    print(f"\nTotal GRIB messages in file: {msg_count}\n")
    print("Unique shortName -> message count:")
    for name, count in sorted(short_names.items()):
        print(f"  {name:12s} : {count} messages")

    print("\nFirst occurrence detail per shortName:")
    for line in details:
        print(line)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python list_grib_messages.py <path_to_grib_file>")
        sys.exit(1)
    list_messages(sys.argv[1])
