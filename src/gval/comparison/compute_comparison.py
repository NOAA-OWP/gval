from typing import Union
from functools import wraps
import inspect
from numbers import Number

# import numpy as np
import numba as nb
import xarray as xr

from gval.comparison.pairing_functions import (
    _make_pairing_dict,
    pairing_dict_fn,
    cantor_pair_signed,
    szudzik_pair_signed,
)
from gval.comparison.agreement import _compute_agreement_map


class ComparisonProcessing:
    """
    Class for Processing Agreement Maps and Tabulations

    Attributes
    ----------
    _func_names : list (private)
        Names of all functions from default categorical statistics class
    _funcs : list (private)
        List of all functions from default categorical statistics class
    _signature_validation : dict (private)
        Dictionary to validate all registered functions
    registered_functions : dict
        Available statistical functions with names as keys and parameters as values
    """

    def __init__(self):
        # Populates default functions for pairing functions
        self._func_names = ["pairing_dict", "cantor", "szudzik"]
        self._funcs = [pairing_dict_fn, cantor_pair_signed, szudzik_pair_signed]

        for name, func in zip(self._func_names, self._funcs):
            setattr(self, name, func)

        self._signature_validation = {
            "names": [],
            "param_types": ["int", "float", "Number"],
            "return_type": [Number],
            "no_of_args": [2, 3],
        }

        self.registered_functions = {
            name: {"params": func_params}
            for name, func_params in zip(
                self._func_names, [["c", "b", "pairing_dict"], ["c", "b"], ["c", "b"]]
            )
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

        return self._signature_validation["names"]

    def register_function(self, name: str, vectorize_func: bool = False):
        """
        Register decorator function in comparison class

        Parameters
        ----------
        name: str
            Name of function to register in comparison class
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
                    nb.vectorize(nopython=True)(func)
                    if vectorize_func is True
                    else func
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
                Class to register pairing functions
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
                        nb.vectorize(nopython=True)(func)
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

    def process_agreement_map(self, **kwargs) -> Union[xr.DataArray, xr.Dataset]:
        """

        Parameters
        ----------
        **kwargs

        Returns
        -------
        Union[xr.DataArray, xr.Dataset]
        Agreement map.
        """

        if isinstance(kwargs["comparison_function"], str):
            if kwargs["comparison_function"] == "pairing_dict":
                if (
                    kwargs.get("pairing_dict") is None
                ):  # this is used for when pairing_dict is not passed
                    # user must set arguments to build pairing dict, throws value error
                    # TODO: consider allow use of unique to acquire all values from candidate and benchmarks
                    if (kwargs.get("allow_candidate_values") is None) | (
                        kwargs.get("allow_benchmark_values") is None
                    ):
                        raise ValueError(
                            "When comparison_function argument is set to 'pairing_dict', must pass values for "
                            "allow_candidate_values and allow_benchmark_values arguments."
                        )

                    # this creates the pairing dictionary from the passed allowed values
                    kwargs["pairing_dict"] = _make_pairing_dict(
                        kwargs.get("allow_candidate_values"),
                        kwargs.get("allow_benchmark_values"),
                    )

            if kwargs["comparison_function"] in self.registered_functions:
                kwargs["comparison_function"] = getattr(
                    self, kwargs["comparison_function"]
                )
            else:
                raise KeyError("Pairing function not found in registered functions")

        return _compute_agreement_map(**kwargs)
