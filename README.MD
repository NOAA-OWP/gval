# gVal: Geospatial Evaluation Framework

gVal (pronounced "g-val") is a high-level Python framework to evaluate
the geospatial skill of candidate maps to benchmarks producing agreement
maps and metrics.

NOTE: THIS PACKAGE IS UNDER ACTIVE DEVELOPMENT AND NOT CURRENTLY
FUNCTIONING. BREAKING CHANGES ARE OFTEN INTRODUCED.

### How to request new features

1.  Visit the Issues page: <https://github.com/NOAA-OWP/gval/issues>.
2.  Click the "New issue" button.
3.  Enter a title that describes the new feature.
4.  Next to "Description", select "Feature Request" in the dropdown box.
5.  Use the template to describe the feature that you would like to
    request.

Please note that all new feature requests must be approved by the
Short-Term Network User Group.

### How to report bugs/issues

1.  Visit the Issues page: <https://github.com/NOAA-OWP/gval/issues>
2.  Click the "New issue" button
3.  Enter a title that describes the bug you have encountered.
4.  Next to "Description", select "Bug" in the dropdown box.
5.  Use the template to describe the bug that you would like to report.

Please note that all bug fixes must be approved by the gVal development
team.

# Installation

## General Use

To install gval in a python environment of choice: <br />
<code>pip install gval</code><br />

# Development Installation

(In this case, the image name, "gval-image", and container name,
"gval-python" can be changed to whatever name is more suitable. Script,
"test.py", does not exist and is an arbitrary placeholder for script of
choice.)

First setup docker instance and in the root directory of the project:
<br /> <code>sudo docker build -t gval-image .</code>

You can either set up your IDE to run this docker image directly:

  - [PyCharm](https://www.jetbrains.com/help/pycharm/using-docker-as-a-remote-interpreter.html#config-docker)
  - [Visual Studio Code](https://code.visualstudio.com/docs/containers/quickstart-python)

Or, running from the command line (from the root directory of the
project): <br /> <code>sudo docker run -v -d $(pwd):/home/user/ --name
gval-python gval-image</code> <br />

Or start up the existing container: <br /> <code>sudo docker start
gval-python</code>

Then run a python file: <code>sudo docker exec gval-python python
test.py</code>

# Packaging

To build the gval package and install locally:

  - In the root directory install the gVal package: <br /> <code>python
    -m build && pip install dist/*.whl --force-reinstall</code>
  - Or on Windows replacing version number with current version number:
    <br /> <code>py -m build</code> then get the name of the wheel in
    the dist directory, e.g.: <br /> <code>pip install
    dist/gval-0.0.1-py3-none-any.whl --force-reinstall</code>

The packaging process using docker container would look as follows (on
linux):

<code>sudo docker exec -v $(pwd):/home/user/ --user root python -m build
&& pip install dist/*.whl --force-reinstall </code>

# Dependencies

## Standards

  - Python 3.10 or higher
  - [Numpy Style Docstrings](https://numpydoc.readthedocs.io/en/v1.1.0/format.html#documenting-modules)
  - [Python Type Hints](https://docs.python.org/3/library/typing.html)
  - [PEP8](https://pep8.org/) Styling (and
    [this](https://peps.python.org/pep-0008/))

## Code Dependencies

  - rioxarray
  - geopandas
  - dask
  - tqdm

## Tooling Dependencies

  - Code Checking with [Flake8](https://github.com/PyCQA/flake8)
      - Combines PyFlakes, pycodestyle, and others for linting and style
        checking.
  - Testing with
    [PyTest](https://docs.pytest.org/en/7.1.x/contents.html)
      - For timed tests:
        [pytest-benchmark](https://pypi.org/project/pytest-benchmark/)
      - For code coverage:
        [pytest-cov](https://pypi.org/project/pytest-cov/)
  - Profiling
      - [yappi](https://github.com/sumerc/yappi/)
      - For memory profiling: [memray](https://pypi.org/project/memray/)
      - Dask memory profiling:
        [dask-memusage](https://github.com/itamarst/dask-memusage)
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

# Contributing

> All contributions to this project will be released to the public
> domain. By submitting a pull request or filing a bug, issue, or
> feature request, you are agreeing to comply with this waiver of
> copyright interest. Details can be found in our [TERMS](07_TERMS.MD)
> and [LICENSE](08_LICENSE.MD).

There are two primary ways to help:

  - Using the issue tracker, and
  - Changing the code-base.

## Using the issue tracker

Use the issue tracker to suggest feature requests, report bugs, and ask
questions. This is also a great way to connect with the developers of
the project as well as others who are interested in this solution.

Use the issue tracker to find ways to contribute. Find a bug or a
feature, mention in the issue that you will take on that effort, then
follow the *Changing the code-base* guidance below.

## Changing the code-base

Generally speaking, you should fork this repository, make changes in
your own fork, and then submit a pull request. All new code should have
associated unit tests that validate implemented features and the
presence or lack of defects. Additionally, the code should follow any
stylistic and architectural guidelines prescribed by the project. In the
absence of such guidelines, mimic the styles and patterns in the
existing code-base.

![alt text](./images/ContributionGraphic.png)

### Guidelines

If you would like to contribute, please follow the following steps:

1.  Fork the project.
2.  Clone your fork: `git clone <your_username>/gval`
3.  Create a feature branch: `git checkout -b <your_feature>`)
4.  Commit your changes: `git commit -m 'feature message'`
5.  Push to the branch: `git push origin <your_username> <your_feature>`
6.  Open a pull request.

# Security Policy

There MUST be no unpatched vulnerabilities of medium or higher severity
that have been publicly known for more than 60 days.

The vulnerability must be patched and released by the project itself
(patches may be developed elsewhere). A vulnerability becomes publicly
known (for this purpose) once it has a CVE with publicly released
non-paywalled information (reported, for example, in the
<a href="https://nvd.nist.gov/">National Vulnerability Database</a>) or
when the project has been informed and the information has been released
to the public (possibly by the project). A vulnerability is considered
medium or higher severity if its
<a href="https://www.first.org/cvss/" >Common Vulnerability Scoring
System (CVSS)</a> base qualitative score is medium or higher. In CVSS
versions 2.0 through 3.1, this is equivalent to a CVSS score of 4.0 or
higher. Projects may use the CVSS score as published in a widely-used
vulnerability database (such as the
<a href="https://nvd.nist.gov">National Vulnerability Database</a>)
using the most-recent version of CVSS reported in that database.
Projects may instead calculate the severity themselves using the latest
version of <a href="https://www.first.org/cvss/">CVSS</a> at the time of
the vulnerability disclosure, if the calculation inputs are publicly
revealed once the vulnerability is publicly known.
<strong>Note</strong>: this means that users might be left vulnerable to
all attackers worldwide for up to 60 days. This criterion is often much
easier to meet than what Google recommends in
<a href="https://security.googleblog.com/2010/07/rebooting-responsible-disclosure-focus.html">Rebooting
responsible disclosure</a>, because Google recommends that the 60-day
period start when the project is notified *even* if the report is not
public. Also note that this badge criterion, like other criteria,
applies to the individual project. Some projects are part of larger
umbrella organizations or larger projects, possibly in multiple layers,
and many projects feed their results to other organizations and projects
as part of a potentially-complex supply chain. An individual project
often cannot control the rest, but an individual project can work to
release a vulnerability patch in a timely way. Therefore, we focus
solely on the individual project's response time. Once a patch is
available from the individual project, others can determine how to deal
with the patch (e.g., they can update to the newer version or they can
apply just the patch as a cherry-picked solution).

The public repositories MUST NOT leak any valid private credential
(e.g., a working password or private key) that is intended to limit
public access.

## Supported Versions

Use this section to tell people about which versions of your project are
currently being supported with security updates.

| Version | Supported            |
| ------- | -------------------- |
| N/a     | :white_check_mark: |
| N/a     | :x:                  |

## Reporting a Vulnerability

Use this section to tell people how to report a vulnerability.

Tell them where to go, how often they can expect to get an update on a
reported vulnerability, what to expect if the vulnerability is accepted
or declined, etc.

# Terms

## Disclaimer

This repository is a scientific product and is not official
communication of the National Oceanic and Atmospheric Administration, or
the United States Department of Commerce. All NOAA GitHub project code
is provided on an 'as is' basis and the user assumes responsibility for
its use. Any claims against the Department of Commerce or Department of
Commerce bureaus stemming from the use of this GitHub project will be
governed by all applicable Federal law. Any reference to specific
commercial products, processes, or services by service mark, trademark,
manufacturer, or otherwise, does not constitute or imply their
endorsement, recommendation or favoring by the Department of Commerce.
The Department of Commerce seal and logo, or the seal and logo of a DOC
bureau, shall not be used in any manner to imply endorsement of any
commercial product or activity by DOC or the United States Government.

[NOAA GitHub Policy](https://github.com/NOAAGov/Information)

### Other Information

Unless expressly stated otherwise, the person who associated a work with
this deed makes no warranties about the work, and disclaims liability
for all uses of the work, to the fullest extent permitted by applicable
law. When using or citing the work, you should not imply endorsement by
the author or the affirmer.

## Exceptions

*Source code or other assets that are excluded from the TERMS should be
listed here. These may include dependencies that may be licensed
differently or are not in the public domain.*

“Software code created by U.S. Government employees is not subject to
copyright in the United States (17 U.S.C. §105). The United
States/Department of Commerce reserve all rights to seek and obtain
copyright protection in countries other than the United States for
Software authored in its entirety by the Department of Commerce. To this
end, the Department of Commerce hereby grants to Recipient a
royalty-free, nonexclusive license to use, copy, and create derivative
works of the Software outside of the United States.”

# Citation

Aristizabal, Fernando and Petrochenkov, Gregory (2023). gVal: Geospatial
Evaluations: National Oceanic and Atmospheric Association, doi:N/A.

    @Manual{
      author = {Fernando Aristizabal,
                Gregory Petrochenkov},
      title = {gVal: Geospatial Evaluation of 2d Datasets},
      publisher = {NOAA},
      address = {Tuscaloosa, AL},
      version = {1.2.0},
      institution = {National Oceananic and Atmosphere Administration},
      year = {2023},
      url = {https://github.com/NOAA-OWP/gval},
    }

##### DOI

N/A

##### Authors

  - [Fernando Aristizabal](https://github.com/fernando-aristizabal):
    Author
  - [Gregory Petrochenkov](https://github.com/GregoryPetrochenkov-NOAA):
    Author
