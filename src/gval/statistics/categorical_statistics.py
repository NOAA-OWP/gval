"""
Categorical Statistics Class
"""

from typing import Union
from functools import wraps
import inspect

import numpy as np
from numba import vectorize

from gval.statistics.base_statistics import BaseStatisticsProcessing
from gval.statistics.categorical_stat_funcs import CategoricalStatistics as cs


class CategoricalStatisticsProcessing(BaseStatisticsProcessing):
    """
    Static Class for Running Categorical Statistics on Agreement Maps

    Attributes
    ----------
    stats : object
        Categorical statistics class to call methods directly
    _func_names : list (private)
        Names of all functions from default categorical statistics class
    _funcs : list (private)
        List of all functions from default categorical statistics class
    _signature_validation : dict (private)
        Dictionary to validate all registered functions
    registered_functions : dict
        Available statistical functions with names as keys and parameters as values
    """

    stats = cs

    # Automatically populates and numba vectorizes all functions in categorical_stat_funcs.py
    _func_names = [fn for fn in dir(cs) if len(fn) > 5 and "__" not in fn]
    _funcs = [getattr(cs, name) for name in _func_names]
    _signature_validation = {
        "names": ["tp", "tn", "fp", "fn"],
        "param_types": ["int", "float", "Number"],
        "return_type": [float],
        "no_of_args": [2, 3, 4],
    }

    # Make all functions first class methods of stats object
    for name, func in zip(_func_names, _funcs):
        setattr(stats, name, vectorize(nopython=True)(func))

    registered_functions = {
        name: {"params": [param for param in inspect.signature(func).parameters]}
        for name, func in zip(_func_names, _funcs)
    }

    @classmethod
    def available_functions(cls) -> list:
        """
        Lists all available functions

        Returns
        -------
        List of available functions
        """
        return list(cls.registered_functions.keys())

    @classmethod
    def get_all_parameters(cls):
        """
        Get all the possible arguments

        Returns
        -------
        List of all possible arguments for functions
        """

        return cls._signature_validation["names"]

    @classmethod
    def register_function(cls, name: str, vectorize_func: bool = False):
        """
        Register decorator function in statistics class

        Parameters
        ----------
        name: str
            Name of function to register in statistics class
        vectorize_func: bool
            Whether to vectorize the function

        Returns
        -------
        Decorator function
        """

        def decorator(func):
            cls.function_signature_check(func)

            if name not in cls.registered_functions:
                cls.registered_functions[name] = {
                    "params": [
                        param
                        for param in inspect.signature(func).parameters
                        if param != "self"
                    ]
                }
                # vectorize funciton if vectorize_func is True
                r_func = (
                    vectorize(nopython=True)(func) if vectorize_func is True else func
                )
                setattr(cls.stats, name, r_func)
            else:
                raise KeyError("This function name already exists")

            @wraps(func)
            def wrapper(*args, **kwargs):  # pragma: no cover
                result = func(*args, **kwargs)

                return result

            return wrapper

        return decorator

    @classmethod
    def register_function_class(cls, vectorize_func: bool = False):
        """
        Register decorator function for an entire class

        Parameters
        ----------
        vectorize_func: bool
            Whether to vectorize the function

        """

        def decorator(dec_cls: object):
            """
            Decorator for wrapper

            Parameters
            ----------
            dec_cls: object
                Class to register stat functions
            """

            for name, func in inspect.getmembers(dec_cls, inspect.isfunction):
                if name not in cls.registered_functions:
                    cls.function_signature_check(func)
                    cls.registered_functions[name] = {
                        "params": [
                            param
                            for param in inspect.signature(func).parameters
                            if param != "self"
                        ]
                    }
                    # vectorize funciton if vectorize_func is True
                    r_func = (
                        vectorize(nopython=True)(func)
                        if vectorize_func is True
                        else func
                    )
                    setattr(cls.stats, name, r_func)
                else:
                    raise KeyError("This function name already exists")

        return decorator

    @classmethod
    def function_signature_check(cls, func):
        """
        Validates signature of registered function

        Parameters
        ----------
        func: function
            Function to check the signature of
        """
        signature = inspect.signature(func)
        names = cls._signature_validation["names"]
        param_types = cls._signature_validation["param_types"]
        return_type = cls._signature_validation["return_type"]
        no_of_args = cls._signature_validation["no_of_args"]

        # Checks if param names, type, and return type are in valid list
        # Considered no validation if either are empty
        for key, val in signature.parameters.items():
            if (key not in names and len(names) > 0) or (
                not str(val).split(": ")[-1] in param_types and len(param_types) > 0
            ):
                raise TypeError(
                    "Wrong parameters in function: \n"
                    f"Valid Names: {names} \n"
                    f"Valid Types: {param_types} \n"
                )

        if len(no_of_args) > 0 and len(signature.parameters) not in no_of_args:
            raise TypeError(
                "Wrong number of parameters: \n"
                f"Valid number of parameters: {no_of_args}"
            )

        if signature.return_annotation not in return_type and len(return_type) > 0:
            raise TypeError("Wrong return type \n" f"Valid return Type {return_type}")

    @classmethod
    def get_parameters(cls, func_name: str) -> list:
        """
        Get parameters of registered function

        Parameters
        ----------
        func_name: str


        Returns
        -------
        List of parameter names for the associated function
        """

        if func_name in cls.registered_functions:
            return cls.registered_functions[func_name]["params"]
        else:
            raise KeyError("Statistic not found in registered functions")

    @classmethod
    def process_statistics(cls, func_names: Union[str, list], arg_dict: dict) -> float:
        """

        Parameters
        ----------
        func_names: Union[str, list]
            Name of registered function to run
        arg_dict: dict
            Dictionary of arguments to pass to function

        Returns
        -------
        Metric from chosen function
        """

        func_names = (
            list(cls.registered_functions.keys()) if func_names == "all" else func_names
        )
        func_list = [func_names] if isinstance(func_names, str) else func_names

        return_stats = []
        for name in func_list:
            if name in cls.registered_functions:
                params = cls.get_parameters(name)
                func = getattr(cls.stats, name)

                # Necessary for numba functions which cannot accept keyword arguments
                func_args = []
                for param in params:
                    if param in arg_dict:
                        func_args.append(arg_dict[param])
                    else:
                        raise ValueError("Parameter missing form kwargs")

                stat_val = func(*func_args)

                if np.isnan(stat_val) or np.isinf(stat_val):
                    raise ValueError(f"Invalid value calculated for {name}:", stat_val)

                return_stats.append(stat_val)

            else:
                raise KeyError("Statistic not found in registered functions")

        return return_stats
