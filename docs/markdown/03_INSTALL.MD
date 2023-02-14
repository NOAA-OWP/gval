# Installation

## General Use

To install gval in a python environment of choice:

`pip install gval`

# Development Installation

(In this case, the image name, "gval-image", and container name, "gval-python" can be changed
to whatever name is more suitable.  Script, "test.py", does not exist and is an arbitrary placeholder for 
script of choice.)

First setup docker instance and in the root directory of the project:

`sudo docker build -t gval-image .`

You can either set up your IDE to run this docker image directly:
- [PyCharm](https://www.jetbrains.com/help/pycharm/using-docker-as-a-remote-interpreter.html#config-docker)
- [Visual Studio Code](https://code.visualstudio.com/docs/containers/quickstart-python)

Or, running from the command line (from the root directory of the project):

`sudo docker run -d -v $(pwd):/home/user/ --name gval-python gval-image`

Or start up the existing container:

`sudo docker start gval-python`

Then run a python file:
`sudo docker exec gval-python python test.py`

# Packaging

To build the gval package and install locally:

- In the root directory install the gVal package:

  `python -m build && pip install dist/*.whl --force-reinstall`
- Or on Windows replacing version number with current version number: 

  `py -m build` then get the name of the wheel in the dist directory, e.g.:

  `pip install dist/gval-0.0.1-py3-none-any.whl --force-reinstall`

The packaging process using docker container would look as follows (on linux):

`sudo docker exec -v $(pwd):/home/user/  --user root python -m build &&
pip install dist/*.whl --force-reinstall `