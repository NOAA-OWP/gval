"""
Automates linting, testing, building, and publishing.

TODO
- [ ] Add static type checking (mypy, pytype, pyre, or pyright)
- [ ] Add version bump
"""

import os
import contextlib

import nox

PYTHON_VERSIONS = ["3.8", "3.9", "3.10", "3.11"]
LINTING_VERSION = "3.11"
DOCS_VERSION = "3.11"

nox.options.default_venv_backend = "conda"


@contextlib.contextmanager
def change_dir(path: str) -> None:
    """
    Change directory using a context manager.
    """
    original_dir = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original_dir)


@nox.session(python=LINTING_VERSION, tags=["linting", "fix"])
def precommit(session: nox.Session) -> None:
    """
    Lint using pre-commit.
    """
    session.install(".[linting,ci_cd_automation]")
    session.run("pre-commit", "run", "--all-files")


'''
@nox.session(python=LINTING_VERSION, tags=["linting"])
def type_check(session: nox.Session) -> None:
    """
    Type check using mypy.
    """
    session.install(".[linting]")
    session.run(
        "mypy", "src", "tests", "noxfile.py", "docs/compile_readme_and_arrange_docs.py"
    )
'''


@nox.session(python=PYTHON_VERSIONS, tags=["testing"])
def tests(session: nox.Session) -> None:
    """
    Run pytests.
    """
    session.install(".[testing]")

    # Run pytests; coverage configuration in .coveragerc and pyproject.toml
    session.run("pytest")


@nox.session(python=LINTING_VERSION, tags=["testing"])
def coverage(session: nox.Session) -> None:
    """
    Report test coverage.
    """
    session.install(".[testing]")

    # Run coverage report; configuration in .coveragerc
    session.run("coverage", "report")


@nox.session(python=DOCS_VERSION, tags=["docs"])
def docs(session: nox.Session) -> None:
    """
    Compile Markdown files and build Sphinx documentation.
    """
    session.install(".[docs]")

    # Define sphinx directory path
    abs_path = os.path.dirname(os.path.abspath(__file__))
    sphinx_dir_path = os.path.join(abs_path, "docs", "sphinx")

    # Change to docs/sphinx folder using the custom context manager
    with change_dir(sphinx_dir_path):
        # run make clean and make html
        session.run("make", "clean", external=True)
        session.run("make", "html", "SPHINXOPTS='-v'", external=True)
