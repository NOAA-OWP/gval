#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gval",
    version="0.0.1",
    author="Fernando Aristizabal",
    author_email="fernando.aristizabal@noaa.gov",
    description="Geospatial Evaluations for a Variety of FIM Outputs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NOAA-OWP/gval",
    packages=setuptools.find_packages("."))