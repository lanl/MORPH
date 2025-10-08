import netCDF4 as nc
import os 
import glob
import argparse

def walk(ncgrp, path="/"):
    # group-level attrs
    if ncgrp.ncattrs():
        print(f"{path} (group) attrs:", dict((k, ncgrp.getncattr(k)) for k in ncgrp.ncattrs()))
    # dimensions
    for dname, dim in ncgrp.dimensions.items():
        print(f"{path}dim {dname}: size={len(dim)}{' (unlimited)' if dim.isunlimited() else ''}")
    # variables
    for vname, var in ncgrp.variables.items():
        print(f"{path}{vname}: shape={var.shape} dtype={var.dtype} dims={var.dimensions}")
        if var.ncattrs():
            print("   attrs:", dict((k, var.getncattr(k)) for k in var.ncattrs()))
    # subgroups (netCDF-4)
    for gname, sub in ncgrp.groups.items():
        walk(sub, path + gname + "/")

parser = argparse.ArgumentParser(description="Explore .nc structure")
parser.add_argument('--data_folder', type=str, default=None, help='data folder')
args = parser.parse_args()
folder = args.data_folder if args.data_folder != None else 'D:/data/pdegym/CE_CRP/'
pattern = os.path.join(folder, '*.nc')
file_paths = sorted(glob.glob(pattern))

for file in file_paths:
    selected_file = file
    print(f'Selected file name: {selected_file}')
    with nc.Dataset(selected_file) as ds:
        walk(ds)