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

If you would like to contribute, please follow the following steps from your terminal within the project's root director:

**Note**: Anything within angle brackets `<>` is meant to be user-specified with the brackets removed.

1. Fork the project on GitHub, clone your fork (`git clone <your_username>/gval`), and create a feature branch (`git checkout -b <your_feature>`).
2. While not required, it's recommended to setup a virtual environment with the Python tool of your choice. Keep in mind that gval supports Python versions 3.8 to 3.11.
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
3. If you setup a virtual environment, make sure to activate it then run `pip install nox`. Alternatively, one could install all the development dependencies with `pip install -e .[dev]`.
4. Set up automatic tasks with pre-commit hooks which are ran prior to every git commit via `pre-commit install`.  This will run [flake8-black](https://pypi.org/project/flake8-black/) to auto-format code and [flake8](https://github.com/PyCQA/flake8) for linting.
5. Code changes: Make your changes and commit them the repo with `git add . && git commit`.
6. Nox: Prior to submitting a PR, run linting, pre-commits, tests, coverage, and doc building with nox. Nox automates various tasks across Python environments.
    - To test across Python versions, you would need to install [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) or [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html). Otherwise, nox will just test in your current Python version.
    - To run all sessions with nox do: `nox`
    - The above could take too long since the tests session is replicated with all Python versions. Try some of the possible variants to reduce computational time:
        - Just precommits: `nox -t precommit`
        - Just tests: `nox -s tests`
        - Just a specific version of Python: `nox -p 3.8`
        - Alternative virtual environment backend: `nox -db mamba`
        - Adding additional Python versions: `nox --extra-pythons 3.7`
    - It's a good idea to preview the docs prior to pushing to remote by opening `docs/sphinx/_build/html/index.html` in your browser.
7. Push to remote with:
    - New branch: `git push -u origin <your_branch>`.
    - Existing branch with upstream set: `git push`.
8. Open a pull request on GitHub and review the checklist in PR template before requesting a review.

#### By-passing Nox
Keep in mind that nox is available as convenience to conduct linting, precommits, testing, coverage, and doc building in an automated fashion. If you choose to do each step manually, please the steps above except for the following steps:
1. Setup a virtual environment if you desire as explained above then within the activated environment run `pip install -e .[dev]`.
2. Run the linters by doing `flake8 src tests noxfile.py` and `black src tests noxfile.py`.
3. Run the tests and coverage by doing `pytest` and ensure 95% test coverage at a minimum.
4. Build the sphinx documentation locally by running `cd docs/sphinx` then do `make clean && make html` to build.


## Standards
- Python 3.8 - 3.11
- [Numpy&nbsp;Style&nbsp;Docstrings](https://numpydoc.readthedocs.io/en/v1.1.0/format.html#documenting-modules)
- [Python&nbsp;Type&nbsp;Hints](https://docs.python.org/3/library/typing.html)
- [PEP8](https://pep8.org/) Styling (and [this](https://peps.python.org/pep-0008/))


## Dependencies
Dependencies are managed with a `requirements.txt` file for core dependencies. A `pyproject.toml` file contains the optional development dependencies and reads the `requirements.txt` file for the core dependencies. Please see those files for a complete list of dependencies used by gval.

## Versioning

The repository will adhere to the [Semantic Versioning 2.0.0.](https://semver.org/)

## Docker Use

A Docker container is available for your convenience.

**Note**: Anything within angle brackets `<>` is meant to be user-specified with the brackets removed. Additionally note that the use of sudo maybe optional dependening on how Docker was setup on your system.

First setup docker instance and in the root directory of the project:

`[sudo] docker build -t <your_image_name> --target development . `

The default user named 'user' with UID 1001 is created.  To use the same user and permissions you
currently have on your machine override with build arguments:

`[sudo] docker build -t <your_image_name> --build-arg UID=$(id -u) --target development .`

Standard run examples from the command line (standard, or overriding user with root):

- `[sudo] docker run -v $(pwd):/home/user/gval --name <your_container_name> <your_image_name>`
- `[sudo] docker run -v $(pwd):/home/user/gval --user root --name <your_container_name> <your_image_name>`

If given access keys for retrieving test data you can mount the volume as such:

`[sudo] docker run -v ~/.aws:/home/user/.aws -v $(pwd):/home/user/gval --name <your_container_name> <your_image_name>`

To keep your container running, try adding `tail -f /dev/null` as your command ensuring to detach with `-d`:
- `[sudo] docker run -d -v $(pwd):/home/user/gval --name <your_container_name> <your_image_name> tail -f /dev/null`

You can also set up your IDE to run this docker image directly:
- [PyCharm](https://www.jetbrains.com/help/pycharm/using-docker-as-a-remote-interpreter.html#config-docker)
- [Visual Studio Code](https://code.visualstudio.com/docs/containers/quickstart-python)

If the container already exists you can start as follows:

`[sudo] docker start <your_container_name>`

To enter the container interactively:
`[sudo] docker exec <your_container_name> bash`
