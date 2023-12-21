# Contributing


> All contributions to this project will be released to the public domain.
> By submitting a pull request or filing a bug, issue, or
> feature request, you are agreeing to comply with this waiver of copyright interest.
> Details can be found in our [TERMS](https://github.com/NOAA-OWP/gval/blob/main/TERMS.MD)
> and [LICENSE](https://github.com/NOAA-OWP/gval/blob/main/LICENSE.MD).


There are two primary ways to help:
- Using the issue tracker, and
- Changing the code-base.


## Using the issue tracker

Use the issue tracker to suggest feature requests, report bugs, and ask questions.
This is also a great way to connect with the developers of the project as well
as others who are interested in this solution.

Use the issue tracker to find ways to contribute. Find a bug or a feature, mention in
the issue that you will take on that effort, then follow the _Changing the code-base_
guidance below.


## Changing the code-base

Generally speaking, you should fork this repository, make changes in your
own fork, and then submit a pull request. All new code should have associated
unit tests that validate implemented features and the presence or lack of defects.
Additionally, the code should follow any stylistic and architectural guidelines
prescribed by the project. In the absence of such guidelines, mimic the styles
and patterns in the existing code-base.

![alt text](https://github.com/NOAA-OWP/gval/raw/main/docs/images/ContributionGraphic.png)

### Guidelines

If you would like to contribute, please follow the following steps from your terminal:

1. Fork the project on GitHub, clone your fork (`git clone <your_username>/gval`), and create a feature branch (`git checkout -b <your_feature>`).
2. Setup nox based linting, testing, and documentation building.
    - Virtual Environment: While not required, it's recommended to setup a virtual environment with the Python tool of your choice. Keep in mind that gval supports Python versions 3.8 to 3.11.
        - With [venv](https://docs.python.org/3/library/venv.html):
            - Ensure your environment is not committed to git. You can use one of the environment names within `.gitignore` to prevent this.
            - To create environment: `python -m venv <your_env_dir>`.
            - To activate: `source <your_env_dir>/bin/activate`.
            - To deactivate: `deactivate`
        - With [conda](https://docs.conda.io/projects/conda/en/latest/index.html):
            - To create: `conda create -n <your_env_name> python=<X.X.X>`.
            - To activate: `conda activate <your_env_name>`.
            - To deactivate: `conda deactivate`.
        - With [mamba](https://mamba.readthedocs.io/en/latest/):
            - Same as conda just use `mamba` instead of `conda`.
    - Development Dependencies: If you setup a virtual environment, make sure to activate it then run `pip install nox`. Alternatively, one could install all the development dependencies with `pip install .[dev]`.
    - Set up pre-commit hooks via `pre-commit install`.  This will run [flake8-black](https://pypi.org/project/flake8-black/) to auto-format code and [flake8](https://github.com/PyCQA/flake8) for linting.
        - 
        - To build sphinx documentation locally, change to the docs/sphinx folder and run `make clean && make html`. The html will be created in the `_build/html` folder.  Open `index.html` in a browser to preview docs.



1. Fork the project on GitHub.
2. Clone your fork: `git clone <your_username>/gval`.
3. Create a feature branch: `git checkout -b <your_feature>`.
4. Setup a virtual environment with a common tool. Keep in mind that gval supports Python versions 3.8 to 3.11. It's recommended to develop in one then use `nox` to test across versions.
    - With [venv](https://docs.python.org/3/library/venv.html):
        - Ensure your environment is not committed to git. You can use one of the environment names within `.gitignore` to prevent this.
        - To create environment: `python -m venv <your_env_dir>`.
        - To activate: `source <your_env_dir>/bin/activate`.
        - To deactivate: `deactivate`
    - With [conda](https://docs.conda.io/projects/conda/en/latest/index.html):
        - To create: `conda create -n <your_env_name> python=<X.X.X>`.
        - To activate: `conda activate <your_env_name>`.
        - To deactivate: `conda deactivate`.
    - With [mamba](https://mamba.readthedocs.io/en/latest/):
        - Same as conda just use `mamba` instead of `conda`.
5. With the environment activated, install the development version of gval via `pip install .[dev]`. This version includes linting, testing, and doc building dependencies.
6. Set up pre-commit hooks via `pre-commit install`.  Afterwards when a commit is made it will run
[flake8](https://github.com/PyCQA/flake8) for code and style linting as well as
[flake8-black](https://pypi.org/project/flake8-black/) to autoformat the code.  In many cases issues will be rectified
automatically with flake8-black but in some cases manual changes will be necessary.
7. The README is made up of other docs located in `/docs/markdown`.  To make changes to the README edit them directly
then run the following script: `python docs/compile_readme_and_arrange_docs.py`.
8. To build sphinx documentation locally, change to the docs/sphinx folder and run `make clean && make html`. The html will be created in the `_build/html` folder.  Open `index.html` in a browser to preview docs.
9. Run linting and tests with nox. Nox automates various tasks across Python environments.
    - To test across Python versions, you would need to install [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) or [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html). Otherwise nox will just test in your current Python version.
    - To run all sessions with nox do: `nox`
    - The above could take too long since the tests session is replicated with all Python versions.
    - Try some of the possible variants to reduce computational time:
        - Just black: `nox -s black`
        - Just flake8: `nox -s flake8`
        - Just tests: `nox -s tests`
        - Just a specific version of Python: `nox -p 3.8`
10. Keep in mind that you can run linting and testing outside of the precommit hooks and nox. Just use the following commands with any custom variation you'd like:
    - For black: `black src/ tests/`.
    - For flake8: `flake8 src/ tests/`
    - For tests: `pytest`
11. Commit your changes: `git commit`. This will invoke pre-commit hooks mentioned on previously.
12. Push to remote with:
    - New branch: `git push -u origin <your_branch>`.
    - Existing branch with upstream set: `git push` .
13. Open a pull request (review checklist in PR template before requesting a review).


## Standards
- Python 3.8 - 3.11
- [Numpy&nbsp;Style&nbsp;Docstrings](https://numpydoc.readthedocs.io/en/v1.1.0/format.html#documenting-modules)
- [Python&nbsp;Type&nbsp;Hints](https://docs.python.org/3/library/typing.html)
- [PEP8](https://pep8.org/) Styling (and [this](https://peps.python.org/pep-0008/))


## Tooling Dependencies
- Testing with [PyTest](https://docs.pytest.org/en/7.1.x/contents.html)
    * For timed tests: [pytest-monitor](https://github.com/CFMTech/pytest-monitor/)
    * For code coverage: [pytest-cov](https://pypi.org/project/pytest-cov/)
    * For memory test: [pytest-memray](https://pytest-memray.readthedocs.io/en/latest/)

- Dependencies and Packaging
    * [pip](https://packaging.python.org/en/latest/key_projects/#pip)
- Distribution
    *[PyPI](https://pypi.org/)
    * GitHub (source, packaging, and images)
- Docs
    * [Sphinx](https://www.sphinx-doc.org/)
    * [pandoc](https://pypi.org/project/pandoc/)
    * [Jupyter](https://pypi.org/project/jupyter/)


## Versioning

The repository will adhere to the [Semantic Versioning 2.0.0.](https://semver.org/)

## Docker Use

(In this case, the image name, "gval-image", and container name, "gval-python" can be changed
to whatever name is more suitable.  Script, "test.py", does not exist and is an arbitrary placeholder for
script of choice.)

First setup docker instance and in the root directory of the project:

`[sudo] docker build -t gval-image --target development . `

The default user named 'user' with UID 1001 is created.  To use the same user and permissions you
currently have on your machine override with build arguments:

`[sudo] docker build -t gval-image --build-arg UID=$(id -u) --target development .`

Standard run examples from the command line (standard, or overriding user with root):

- `[sudo] docker run -v $(pwd):/home/user/gval --name gval-python gval-image`
- `[sudo] docker run -v $(pwd):/home/user/gval --user root --name gval-python gval-image`

If given access keys for retrieving test data you can mount the volume as such:

`[sudo] docker run -v ~/.aws:/home/user/.aws -v $(pwd):/home/user/gval --name gval-python gval-image`

To keep your container running, try adding `tail -f /dev/null` as your command ensuring to detach with `-d`:
- `[sudo] docker run -d -v $(pwd):/home/user/gval --name gval-python gval-image tail -f /dev/null`

You can also set up your IDE to run this docker image directly:
- [PyCharm](https://www.jetbrains.com/help/pycharm/using-docker-as-a-remote-interpreter.html#config-docker)
- [Visual Studio Code](https://code.visualstudio.com/docs/containers/quickstart-python)

If the container already exists you can start as follows:

`[sudo] docker start gval-python`

To enter the container interactively:
`[sudo] docker exec gval-python bash`

## Packaging

To build the gval package and install locally:

- In the root directory install the gval package:

  `pip install . .[dev]`

The packaging process using docker container would look as follows (on linux):

`sudo docker exec -v $(pwd):/home/user/ --user root python -m build && pip install . .[dev]`
