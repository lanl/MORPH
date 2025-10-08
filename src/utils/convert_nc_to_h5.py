# convert_nc_to_h5.py
import netCDF4 as nc
import h5py
import argparse
import os
import glob

def copy_attrs(ncobj, h5obj):
    for name in ncobj.ncattrs():
        h5obj.attrs[name] = ncobj.getncattr(name)

def copy_var(ncvar, h5group):
    # Try to preserve chunking/compression if present
    chunks = None
    try:
        ch = ncvar.chunking()
        if isinstance(ch, tuple):
            chunks = ch
    except Exception:
        pass
    compression = None
    shuffle = False
    try:
        f = ncvar.filters()
        if f.get("zlib"):
            compression = "gzip"
        shuffle = bool(f.get("shuffle", False))
    except Exception:
        pass
    dset = h5group.create_dataset(
        ncvar.name.replace("/", "_"),
        data=ncvar[...],
        chunks=chunks,
        compression=compression,
        shuffle=shuffle,
    )
    copy_attrs(ncvar, dset)

def copy_group(ncgrp, h5grp):
    copy_attrs(ncgrp, h5grp)
    # Save dimension sizes as attributes (optional)
    for dim_name, dim in ncgrp.dimensions.items():
        h5grp.attrs[f"dim_{dim_name}"] = -1 if dim.isunlimited() else len(dim)
    for name, var in ncgrp.variables.items():
        copy_var(var, h5grp)
    for name, subgrp in ncgrp.groups.items():
        copy_group(subgrp, h5grp.create_group(name))

def convert_nc_to_h5(nc_path, h5_path):
    with nc.Dataset(nc_path, "r") as src, h5py.File(h5_path, "w") as dst:
        copy_group(src, dst)

parser = argparse.ArgumentParser(description="convert .nc to .h5 structure")
parser.add_argument('--source_nc_loc', type=str, required = True, help='source folder')
parser.add_argument('--target_h5_loc', type=str, required = True, help='target folder')
args = parser.parse_args()
source_files = os.path.join(args.source_nc_loc, '*.nc')
source_files = sorted(glob.glob(source_files))
print(f'Total files = {len(source_files)} inside {args.source_nc_loc} \n All .nc files -> {source_files} ')

for file in source_files:
    filename = os.path.splitext(os.path.basename(file))[0]
    target_filename = filename + '.h5'
    target_address = os.path.join(args.target_h5_loc, target_filename)
    print(f"Converting {file} to {target_address}...")
    convert_nc_to_h5(file, target_address)
