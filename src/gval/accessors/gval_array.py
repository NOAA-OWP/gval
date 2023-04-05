import xarray as xr

from gval.accessors.gval_xarray import GVALXarray


@xr.register_dataarray_accessor("gval")
class GVALArray(GVALXarray):
    """
    Class for extending xarray DataArray functionality

    Attributes
    ----------
    _obj : xr.DataArray
       Object to use off the accessor
    data_type : type
        Data type of _obj
    """

    def __init__(self, xarray_obj: xr.DataArray):
        super().__init__(xarray_obj)
