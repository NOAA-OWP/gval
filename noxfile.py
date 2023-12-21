"""
Automates linting, testing, building, and publishing.

TODO
- [ ] Add docs build
- [ ] Add static type checking (mypy, pytype, pyre, or pyright)
- [ ] Add version bump
- [ ] Add publish to pypi
"""

import os
import contextlib

import nox

MIN_COVERAGE_PERC = 85
PYTHON_VERSIONS = ["3.8", "3.9", "3.10", "3.11"]
LINTING_VERSION = "3.11"
DOCS_VERSION = "3.10"

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
def black(session: nox.Session) -> None:
    """
    Format code using flake8-black.
    """
    session.install(".[linting]")
    session.run("black", "src", "tests", "noxfile.py", "docs/compile_readme_and_arrange_docs.py", "--check")


@nox.session(python=LINTING_VERSION, tags=["linting"])
def flake8(session: nox.Session) -> None:
    """
    Lint using flake8.
    """
    session.install(".[linting]")
    session.run("flake8", "src", "tests", "noxfile.py", "docs/compile_readme_and_arrange_docs.py")


@nox.session(python=PYTHON_VERSIONS, tags=["testing"])
def tests(session: nox.Session) -> None:
    """
    Run pytests.
    """
    session.install(".[testing]")
    session.run(
        "pytest",
        "--memray",
        "--cov=gval",
        "--cov-report",
        "term-missing",
        f"--cov-fail-under={MIN_COVERAGE_PERC}",
    )


'''
@nox.session(python=PYTHON_VERSIONS, tags=["testing"])
def coverage(session: nox.Session) -> None:
    """
    Run pytests.
    """
    session.install(".[testing]")
    session.run("coverage", "report", f"--fail-under={MIN_COVERAGE_PERC}")
'''

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
