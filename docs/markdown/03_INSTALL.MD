# Installation

## General Use

To install gval in a python environment of choice: <br />
<code>pip install gval</code><br />

# Development Installation

(In this case, the image name, "gval-image", and container name, "gval-python" can be changed
to whatever name is more suitable.  Script, "test.py", does not exist and is an arbitrary placeholder for 
script of choice.)

First setup docker instance and in the root directory of the project: <br />
<code> sudo docker build -t gval-image .</code>

You can either set up your IDE to run this docker image directly:
- [PyCharm](https://www.jetbrains.com/help/pycharm/using-docker-as-a-remote-interpreter.html#config-docker)
- [Visual Studio Code](https://code.visualstudio.com/docs/containers/quickstart-python)

Or, running from the command line (from the root directory of the project): <br />
<code> sudo docker run -v $(pwd):/home/user/ --name gval-python gval-image</code> <br />

Or start up the existing container: <br />
<code> sudo docker start gval-python </code>

Then run a python file:
<code> sudo docker exec gval-python /usr/local/gval_env/bin/python test.py</code>

# Packaging

To build the gval package and install locally:

- In the root directory install the gVal package: <br />
  <code>python -m build && pip install dist/*.whl --force-reinstall</code>
- Or on Windows replacing version number with current version number: <br />
  <code>py -m build</code> then get the name of the wheel in the dist directory, e.g.: <br />
  <code>pip install dist/gval-0.0.1-py3-none-any.whl --force-reinstall</code>

The packaging process using docker container would look as follows (on linux):

<code>sudo docker exec -v $(pwd):/home/user/  --user root gval-image 
/usr/local/gval_env/bin/python -m build &&
/usr/local/gval_env/bin/pip install dist/*.whl --force-reinstall </code>