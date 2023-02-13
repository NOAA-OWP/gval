# Dependencies

## Standards
- Python 3.10 or higher
- [Numpy&nbsp;Style&nbsp;Docstrings](https://numpydoc.readthedocs.io/en/v1.1.0/format.html#documenting-modules)
- [Python&nbsp;Type&nbsp;Hints](https://docs.python.org/3/library/typing.html)
- [PEP8](https://pep8.org/) Styling (and [this](https://peps.python.org/pep-0008/))

## Code Dependencies
- rioxarray
- geopandas
- dask
- tqdm

## Tooling Dependencies
- Code Checking with [Flake8](https://github.com/PyCQA/flake8)
    - Combines PyFlakes, pycodestyle, and others for linting and style checking.
- Testing with [PyTest](https://docs.pytest.org/en/7.1.x/contents.html)
    - For timed tests: [pytest-benchmark](https://pypi.org/project/pytest-benchmark/)
    - For code coverage: [pytest-cov](https://pypi.org/project/pytest-cov/)
- Profiling
    - [yappi](https://github.com/sumerc/yappi/)
    - For memory profiling: [memray](https://pypi.org/project/memray/)
    - Dask memory profiling: [dask-memusage](https://github.com/itamarst/dask-memusage)
- Dependencies and Packaging
    - [pip](https://packaging.python.org/en/latest/key_projects/#pip)
    - Docker
- Distribution
    - [PyPI](https://pypi.org/)
    - DockerHub
    - GitHub (source, packaging, and images)
- Docs
    - [Sphinx](https://www.sphinx-doc.org/)
    - [pandoc](https://pypi.org/project/pandoc/)
    - [ReadTheDocs](https://readthedocs.org/)
    - [Jupyter](https://pypi.org/project/jupyter/)