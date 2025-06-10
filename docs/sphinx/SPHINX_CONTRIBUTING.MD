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

Your contributing guidelines are well-structured and detailed, providing clear steps for contributors. Here are some suggestions to enhance clarity, conciseness, and precision:

### Guidelines

If you would like to contribute to our project, please follow these steps from the terminal within the project's root directory:

**Note**: Replace anything within angle brackets `<>` with your specific information, removing the brackets.

1. **Fork and Clone**:
   - Fork the project on GitHub.
   - Clone your fork (`git clone <your_username>/gval`).
   - Create a feature branch (`git checkout -b <your_feature>`).

2. **Virtual Environment Setup** (Recommended):
   - **gval** supports Python versions 3.8 to 3.11.
   - Ensure your environment is not included in the git commits. Use one of the names listed in `.gitignore`.
   - Options:
     - **venv**:
       - Create: `python -m venv <your_env_dir>`.
       - Activate: `source <your_env_dir>/bin/activate`.
       - Deactivate: `deactivate`.
     - **conda/mamba**:
       - Create: `[conda|mamba] create -n <your_env_name> python=<X.X.X>`.
       - Activate: `[conda|mamba] activate <your_env_name>`.
       - Deactivate: `[conda|mamba] deactivate`.

3. **Dependency Installation**:
   - Activate your virtual environment.
   - Install Nox: `pip install nox`.
   - Alternatively, install all development dependencies: `pip install -e .[dev]`.

4. **Pre-commit Hooks**:
   - Set up automatic linting tasks: `pre-commit install`.

5. **Making Changes**:
   - Commit your changes: `git add . && git commit`.

6. **Using Nox**:
   - Run linting, pre-commits, tests, coverage, and doc building with Nox.
   - To test across Python versions, install [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) or [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).
   - Run all sessions (time-consuming): `nox`.
   - Variants to save time:
     - Pre-commits only (linting): `nox -t precommit`.
     - Tests only: `nox -s tests`.
     - Specific Python version: `nox -p 3.8`.
     - Different virtual environment backend: `nox -db mamba`.
     - Extra Python versions: `nox --extra-pythons 3.7`.
   - Preview docs after running `nox` or `nox -s docs`: Open `docs/sphinx/_build/html/index.html` in a browser.

7. **Pushing Changes**:
   - New branch: `git push -u origin <your_branch>`.
   - Existing branch: `git push`.

8. **Pull Request**:
    - Start a PR conforming to the PR template and request a review.

#### By-passing Nox

You may manually perform steps handled by Nox:
1. Setup a virtual environment (as above) and run `pip install -e .[dev]`.
2. Linting: Execute `flake8` and `black .`.
3. Testing: Run `pytest` ensuring at least 95% test coverage.
4. Docs: Navigate (`cd docs/sphinx`) and build (`make clean && make html`).


## Dependencies
- **Core dependencies** are managed within `requirements.txt`.
- **Optional dependencies** related to development activities are listed directly within `pyproject.toml`.
    - There are several groups of optional dependencies but `dev` encompasses all of them.

## Versioning

The repository adheres to the [Semantic Versioning 2.0.0.](https://semver.org/)

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
