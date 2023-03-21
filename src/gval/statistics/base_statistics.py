"""
Base Statistics Class
"""

from abc import ABC, abstractmethod


class BaseStatistics(ABC):  # pragma: no cover
    """
    Abstract Class for Running Statistics on Agreement Maps

    Attributes
    ----------
    registered_functions : dict
        Available statistical functions for respective statistics class
    signature_validation: dict
        Dictionary with parameter name and types to constrain registered functions
    """

    def __init__(self):
        self.registered_functions = {}
        self.signature_validation = {}

    @abstractmethod
    def register_function(self, name: str):
        """
        Registers function in statistics class

        Parameters
        ----------
        name: str
            Name of function to register in statistics class
        """
        pass

    @abstractmethod
    def available_functions(self) -> list:
        """
        Lists all available functions

        Returns
        -------
        List of available functions
        """
        return self.registered_functions.keys()

    @abstractmethod
    def function_signature_check(self, func):
        """
        Validates signature of registered function

        Returns
        -------
        List of available functions
        """
        pass
