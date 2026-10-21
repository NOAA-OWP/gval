import lazy_loader as _lazy

_lazy_getattr, _, __all__ = _lazy.attach_stub(__name__, __file__)

name_dict = {
    "Comparison": "ComparisonProcessing",
    "CatStats": "CategoricalStatistics",
    "ContStats": "ContinuousStatistics",
}


def __getattr__(name):
    if name in name_dict:
        name = name_dict[name]

    obj = _lazy_getattr(name)

    match name:
        case "ComparisonProcessing":
            if "Comparison" not in globals():
                globals()["Comparison"] = obj()
            return obj
        case "CategoricalStatistics":
            if "CatStats" not in globals():
                globals()["CatStats"] = obj()
            return obj
        case "ContinuousStatistics":
            if "ContStats" not in globals():
                globals()["ContStats"] = obj()
            return obj
        case _:
            return obj


from gval.accessors import gval_array, gval_dataset, gval_dataframe
