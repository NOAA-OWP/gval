"""
Package-level configuration file.
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal, Gregory Petrochenkov"


import os


# defines absolute path of project's root directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    print("Docker works from cmd")
