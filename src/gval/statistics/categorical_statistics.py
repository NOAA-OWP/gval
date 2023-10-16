"""
Categorical Statistics Class
"""

from typing import Union, Tuple
from functools import wraps
import inspect

import numpy as np
from numba import vectorize

from gval.statistics.base_statistics import BaseStatistics
import gval.statistics.categorical_stat_funcs as cs


class CategoricalStatistics(BaseStatistics):
    """
    Class for Running Categorical Statistics on Agreement Maps

    Attributes
    ----------
    registered_functions : dict
        Available statistical functions with names as keys and parameters as values
    """

    def __init__(self):
        # Automatically populates and numba vectorizes all functions in categorical_stat_funcs.py

        self.required_param = 1
        self.optional_param = 0

        self._func_names = [
            fn
            for fn in dir(cs)
            if len(fn) > 5 and "__" not in fn and "Number" not in fn
        ]
        self._funcs = [getattr(cs, name) for name in self._func_names]

        for name, func in zip(self._func_names, self._funcs):
            setattr(self, name, vectorize(nopython=True)(func))

        self._signature_validation = {
            "names": {
                "tp": self.required_param,
                "tn": self.optional_param,
                "fp": self.required_param,
                "fn": self.required_param,
            },
            "required": [
                self.required_param,
                self.optional_param,
                self.required_param,
                self.required_param,
            ],
            "param_types": ["int", "float", "Number"],
            "return_type": [float],
            "no_of_args": [2, 3, 4],
        }

        self.registered_functions = {
            name: {"params": [param for param in inspect.signature(func).parameters]}
            for name, func in zip(self._func_names, self._funcs)
        }

    def available_functions(self) -> list:
        """
        Lists all available functions

        Returns
        -------
        List of available functions
        """
        return list(self.registered_functions.keys())

    def get_all_parameters(self):
        """
        Get all the possible arguments

        Returns
        -------
        List of all possible arguments for functions
        """

        return list(self._signature_validation["names"].keys())

    def register_function(self, name: str, vectorize_func: bool = False):
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
            self.function_signature_check(func)

            if name not in self.registered_functions:
                self.registered_functions[name] = {
                    "params": [
                        param
                        for param in inspect.signature(func).parameters
                        if param != "self"
                    ]
                }
                # vectorize function if vectorize_func is True
                r_func = (
                    vectorize(nopython=True)(func) if vectorize_func is True else func
                )
                setattr(self, name, r_func)
            else:
                raise KeyError("This function name already exists")

            @wraps(func)
            def wrapper(*args, **kwargs):  # pragma: no cover
                result = func(*args, **kwargs)

                return result

            return wrapper

        return decorator

    def register_function_class(self, vectorize_func: bool = False):
        """
        Register decorator function for an entire class

        Parameters
        ----------
        vectorize_func: bool
            Whether to vectorize the function

        """

        def decorator(dec_self: object):
            """
            Decorator for wrapper

            Parameters
            ----------
            dec_self: object
                Class to register stat functions
            """

            for name, func in inspect.getmembers(dec_self, inspect.isfunction):
                if name not in self.registered_functions:
                    self.function_signature_check(func)
                    self.registered_functions[name] = {
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
                    setattr(self, name, r_func)
                else:
                    raise KeyError("This function name already exists")

        return decorator

    def function_signature_check(self, func):
        """
        Validates signature of registered function

        Parameters
        ----------
        func: function
            Function to check the signature of
        """
        signature = inspect.signature(func)
        names = self._signature_validation["names"]
        param_types = self._signature_validation["param_types"]
        return_type = self._signature_validation["return_type"]
        no_of_args = self._signature_validation["no_of_args"]

        # Checks if param names, type, and return type are in valid list
        # Considered no validation if either are empty
        for key, val in signature.parameters.items():
            if (key not in names and len(names) > 0) or (
                not str(val).split(": ")[-1].split(".")[-1] in param_types
                and len(param_types) > 0
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

    def get_parameters(self, func_name: str) -> list:
        """
        Get parameters of registered function

        Parameters
        ----------
        func_name: str


        Returns
        -------
        List of parameter names for the associated function
        """

        if func_name in self.registered_functions:
            return self.registered_functions[func_name]["params"]
        else:
            raise KeyError("Statistic not found in registered functions")

    def process_statistics(
        self, func_names: Union[str, list], **kwargs
    ) -> Tuple[float, str]:
        """

        Parameters
        ----------
        func_names: Union[str, list]
            Name of registered function to run
        **kwargs: dict or keyword arguments
            Dictionary or keyword arguments of to pass to metric functions.

        Returns
        -------
        Tuple[float, str]
            Tuple with metric values and metric names.
        """

        func_names = (
            list(self.registered_functions.keys())
            if func_names == "all"
            else func_names
        )
        func_list = [func_names] if isinstance(func_names, str) else func_names

        return_stats, return_funcs = [], []
        for name in func_list:
            if name in self.registered_functions:
                params = self.get_parameters(name)
                required = self._signature_validation["required"]

                func = getattr(self, name)

                # Necessary for numba functions which cannot accept keyword arguments
                func_args, skip_function, return_nan = [], False, False
                for param, req in zip(params, required):
                    if param in kwargs:
                        func_args.append(kwargs[param])
                    elif not self._signature_validation["names"][param]:
                        skip_function = True
                        break
                    else:
                        return_nan = True
                        break

                if skip_function:
                    continue

                with np.errstate(divide="ignore", invalid="ignore"):
                    stat_val = np.nan if return_nan else func(*func_args)

                return_stats.append(stat_val)
                return_funcs.append(name)

            else:
                raise KeyError(f"Statistic, {name}, not found in registered functions")

        return return_stats, return_funcs
