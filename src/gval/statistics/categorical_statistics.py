"""
Categorical Statistics Class
"""

from functools import wraps
import inspect

import numpy as np
from numba import vectorize

from gval.statistics.base_statistics import BaseStatistics
import gval.statistics.categorical_stat_funcs as cat_stats


class CategoricalStatistics(BaseStatistics):
    """
    Static Class for Running Categorical Statistics on Agreement Maps

    Attributes
    ----------
    registered_functions : dict
        Available statistical functions for respective statistics class
    signature_validation: dict
        Dictionary with parameter name and types to constrain registered functions
    """

    # Automatically populates and numba vectorizes all functions in categorical_stat_funcs.py
    func_names = [fn for fn in dir(cat_stats) if len(fn) > 5 and "__" not in fn]
    funcs = [getattr(cat_stats, name) for name in func_names]
    registered_functions = {
        name: {
            "params": [param for param in inspect.signature(func).parameters],
            "func": vectorize(nopython=True)(getattr(cat_stats, name)),
        }
        for name, func in zip(func_names, funcs)
    }
    signature_validation = {
        "names": ["tp", "tn", "fp", "fn"],
        "param_types": ["int", "float", "Number"],
        "return_type": [float],
        "no_of_args": [2, 3, 4],
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

        return cls.signature_validation["names"]

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
                    ],
                    "func": vectorize(nopython=True)(func)
                    if vectorize_func is True
                    else func,
                }
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
                        ],
                        "func": vectorize(nopython=True)(func)
                        if vectorize_func is True
                        else func,
                    }  # pragma: no cover
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
        names = cls.signature_validation["names"]
        param_types = cls.signature_validation["param_types"]
        return_type = cls.signature_validation["return_type"]
        no_of_args = cls.signature_validation["no_of_args"]

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
    def process_statistic(cls, func_name: str, arg_dict: dict) -> float:
        """

        Parameters
        ----------
        func_name: str
            Name of registered function to run
        arg_dict: dict
            Dictionary of arguments to pass to function

        Returns
        -------
        Metric from chosen function
        """

        if func_name in cls.registered_functions:
            params = cls.registered_functions[func_name]["params"]
            func = cls.registered_functions[func_name]["func"]

            # Necessary for numba functions which cannot accept keyword arguments
            func_args = []
            for param in params:
                if param in arg_dict:
                    func_args.append(arg_dict[param])
                else:
                    raise ValueError("Parameter missing form kwargs")

            ret_val = func(*func_args)

            if np.isnan(ret_val) or np.isinf(ret_val):
                raise ValueError(f"Invalid value calculated for {func_name}:", ret_val)

            return ret_val

        else:
            raise KeyError("Statistic not found in registered functions")
