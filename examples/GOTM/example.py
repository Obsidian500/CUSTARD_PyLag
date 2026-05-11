import xarray as xr

ds = xr.load_dataset("/home/banga/repos/PyLag/CUSTARD_PyLag/examples/GOTM/data/gotm_l4_20_level.nc")
print(ds)