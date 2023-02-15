FROM python:3.10 AS builder
SHELL ["/bin/bash", "-c"]
# TRICK TO USE DIFFERENT PYTHON VERSIONS
#ARG PYTHON_VERSION=3.7.0-alpine3.8
#FROM python:${PYTHON_VERSION} as builder

# TRY ALPINE LINUX PYTHON FOR SMALLER IMAGE
#FROM python:3.10-rc-alpine3.16 AS builder

## ARGS ##
ARG REQS=base
ARG DEVREQS=test
ARG VENV=/usr/local/gval_env
ARG PROJDIR=/home/user/gval
ARG VERSION='0.0.1'
ARG MAINTANER='Fernando Aristizabal'
ARG RELEASE_DATE=''

## SETUP ENV VARS ##
ENV VENV=$VENV
ENV PROJDIR=$PROJDIR

## COPY IN REQUIREMENTS ##
COPY requirements/$REQS.txt /tmp
COPY requirements/$DEVREQS.txt /tmp


## INSTALL EXTERNAL DEPENDENCIES ##
# remove versions if errors occur
RUN python3 -m venv $VENV && \
    $VENV/bin/pip install --upgrade build && \
    $VENV/bin/pip install -r /tmp/$REQS.txt && \
    $VENV/bin/pip install -r /tmp/$DEVREQS.txt && \
    rm -rf /tmp/*

# If we want the GDAL python dep we need this
#RUN $VENV/bin/pip install setuptools==57.5.0 && \
#   $VENV/bin/pip install GDAL==3.2.2

# TRY USING $VENV/bin/pip???
#RUN $VENV/bin/pip install -r /tmp/$REQS.txt && \
#    rm -rf /tmp/*

###############################################################################################
# development stage
###############################################################################################
FROM python:3.10 AS development

## ARGS ##
ARG REQS=base
ARG DEVREQS=test
ARG VENV=/usr/local/gval_env
ARG PROJDIR=/home/user/gval
ARG VERSION='0.0.1'
ARG MAINTANER='Fernando Aristizabal'
ARG RELEASE_DATE=''

# Label docker image
LABEL version=$VERSION \
      maintaner=$MAINTANER \
      release-date=$RELEASE_DATE

## SETTING ENV VARIABLES ##
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
# ensures stdout stderr are sent straight to terminal
ENV PYTHONUNBUFFERED=TRUE 
ENV VENV=$VENV
ENV PROJDIR=$PROJDIR
# set path to virtual env so that future python commands use it
ENV PATH="$VENV/bin:$PATH"

# RETRIEVE BUILT DEPENDENCIES
COPY --from=builder $VENV $VENV



## ADDING USER GROUP ##
ARG UID=1001
ARG UNAME=user
RUN useradd -Ums /bin/bash -u $UID $UNAME
USER $UNAME
WORKDIR /home/$UNAME

## ADDING ALIASES TO USER'S BASH ALIASES FILE ##
#RUN echo 'alias python="$VENV/bin/python3"' >> /home/$UNAME/.bash_aliases
#RUN echo 'alias pip="$VENV/bin/pip"' >> /home/$UNAME/.bash_aliases

## ENTRYPOINT: infinitely tails nothing to keep container alive
ENTRYPOINT ["tail", "-f", "/dev/null"]

###############################################################################################
# runtime stage
###############################################################################################
#FROM development AS runtime

#COPY . $PROJDIR
#WORKDIR $PROJDIR
#RUN $VENV/bin/pip install $PROJDIR

#CMD ["./.venv/bin/python", "-m", "$PROJDIR/main.py"]
