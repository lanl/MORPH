import argparse
import os
from pathlib import Path
import glob
import netCDF4 as nc
import numpy as np
import h5py
import sys

def _safe_name(parts):
    return "__".join(parts).replace("/", "_")

def _to_plain_ndarray(var_or_ds):
    arr = var_or_ds[...]
    # handle masked arrays (NetCDF)
    if np.ma.isMaskedArray(arr):
        if np.issubdtype(arr.dtype, np.number):
            arr = arr.filled(np.nan)
        elif arr.dtype.kind in ("S", "U"):
            arr = arr.filled(b"" if arr.dtype.kind == "S" else "")
        else:
            arr = arr.filled(0)
    # classic NetCDF char arrays -> strings, then decode bytes
    if arr.dtype.kind == "S":
        try:
            if arr.dtype.itemsize == 1:
                arr = nc.chartostring(arr, axis=-1)
            arr = np.char.decode(arr, "utf-8", "replace")
        except Exception:
            pass
    # HDF5 vlen/object strings
    if arr.dtype == object:
        try:
            flat = [
                (x.decode("utf-8", "replace") if isinstance(x, (bytes, bytearray))
                 else ("" if x is None else str(x)))
                for x in arr.ravel()
            ]
            arr = np.array(flat, dtype="U").reshape(arr.shape)
        except Exception:
            pass
    return arr

def convert_nc_to_npy(nc_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(nc_path, "r") as ds:
        def walk(grp, prefix):
            for vname, var in grp.variables.items():
                arr = _to_plain_ndarray(var)
                name = nc_path.stem
                np.save(out_dir / f"{name}.npy", arr)
            for gname, sub in grp.groups.items():
                walk(sub, prefix + [gname])
        walk(ds, [])

def convert_h5_to_npy(h5_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "r") as h5f:
        def walk(grp, prefix):
            for name, item in grp.items():
                if isinstance(item, h5py.Dataset):
                    try:
                        arr = _to_plain_ndarray(item)
                        np.save(out_dir / (_safe_name(prefix + [name]) + ".npy"), arr)
                    except Exception as e:
                        print(f"skip {('/'.join(prefix+[name]))}: {e}")
                elif isinstance(item, h5py.Group):
                    walk(item, prefix + [name])
        walk(h5f, [])

def main():
    p = argparse.ArgumentParser(description="Convert .nc or .h5/.hdf5 to per-dataset .npy files")
    p.add_argument("--data_format", choices=["nc", "h5"], default="nc", 
                   help="input data format")
    p.add_argument("--source_nc_file", help="folder with .nc/.nc4 files")
    p.add_argument("--source_h5_file", help="folder with .h5/.hdf5 files")
    p.add_argument("--target_npy_loc", required=True, help="output folder for .npy")
    args = p.parse_args()

    # Project root
    project_root = Path(__file__).resolve().parent.parent.parent
    dataset_root = project_root / "datasets"
    os.makedirs(dataset_root, exist_ok=True)
    dst_dir = dataset_root / Path(args.target_npy_loc)
    print(f"Output .npy files will be saved to: {dst_dir}")

    if args.data_format == "nc":
        if not args.source_nc_file:
            p.error("--source_nc_file is required when --data_format=nc")
        src_file = dataset_root / Path(args.source_nc_file)
        print(f'Loading .nc files from {src_file}')
        print(f"Saving arrays from {src_file} -> {dst_dir}")
        convert_nc_to_npy(src_file, dst_dir)

    else:  # h5
        if not args.source_h5_file:
            p.error("--source_h5_file is required when --data_format=h5")
        src_file = dataset_root / Path(args.source_h5_file)
        print(f'Loading .h5/.hdf5 files from {src_file}')
        print(f"Saving arrays from {src_file} -> {dst_dir}")
        convert_h5_to_npy(src_file, dst_dir)

if __name__ == "__main__":
    main()
