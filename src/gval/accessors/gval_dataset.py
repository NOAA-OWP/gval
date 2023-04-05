import xarray as xr

from gval.accessors.gval_xarray import GVALXarray


@xr.register_dataset_accessor("gval")
class GVALDataset(GVALXarray):
    """
    Class for extending xarray Dataset functionality

    Attributes
    ----------
    _obj : xr.Dataset
       Object to use off the accessor
    data_type : type
        Data type of _obj
    """

    def __init__(self, xarray_obj: xr.Dataset):
        super().__init__(xarray_obj)
