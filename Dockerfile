FROM python:3.10 AS builder

# TRICK TO USE DIFFERENT PYTHON VERSIONS
#ARG PYTHON_VERSION=3.7.0-alpine3.8
#FROM python:${PYTHON_VERSION} as builder

# TRY ALPINE LINUX PYTHON FOR SMALLER IMAGE
#FROM python:3.10-rc-alpine3.16 AS builder

## ARGS ##
ARG REQS=base
ARG DEVREQS=test
ARG VENV=/usr/local/gval_env
ARG PROJDIR=/gval
ARG VERSION=''
ARG MAINTANER='Fernando Aristizabal'
ARG RELEASE_DATE=''

## SETUP ENV VARS ##
ARG VENV=$VENV
ARG PROJDIR=$PROJDIR

## COPY IN REQUIREMENTS ##
COPY requirements/$REQS.txt /tmp
COPY requirements/$DEVREQS.txt /tmp

## INSTALL EXTERNAL DEPENDENCIES ##
# remove versions if errors occur
RUN apt update --fix-missing && \
    DEBIAN_FRONTEND=noninteractive \
        apt install -qy \
            gdal-bin=3.2.2+dfsg-2+deb11u2 \
            libgdal-dev=3.2.2+dfsg-2+deb11u2 \
            python3-gdal=3.2.2+dfsg-2+deb11u2 && \
            pandoc=2.9.2.1 && \
    apt auto-remove -y && \
    python3 -m venv $VENV && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/* && \
    $VENV/bin/pip install --upgrade build && \
    $VENV/bin/pip install -r /tmp/$REQS.txt && \
    $VENV/bin/pip install -r /tmp/$DEVREQS.txt && \
    rm -rf /tmp/*

# If we want the GDAL python dep we need this
RUN $VENV/bin/pip install setuptools==57.5.0 && \
    $VENV/bin/pip install GDAL==3.2.2

# TRY USING $VENV/bin/pip???
#RUN $VENV/bin/pip install -r /tmp/$REQS.txt && \
#    rm -rf /tmp/*

###############################################################################################
# development stage
###############################################################################################
FROM python:3.10 AS development

## SETTING ENV VARIABLES ##
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
# ensures stdout stderr are sent straight to terminal
ENV PYTHONUNBUFFERED=TRUE 

## Virtual and project directories ##
#ARG VENV=$VENV
#ARG PROJDIR=$PROJDIR

# Label docker image
LABEL version=$VERSION \
      maintaner=$MAINTANER \
      release-date=$RELEASE_DATE

# RETRIEVE BUILT DEPENDENCIES
COPY --from=builder $VENV $VENV

# set path to virtual env so that future python commands use it
ENV PATH="$VENV/bin:$PATH"

## ADDING USER GROUP ##
ARG UID=1001
ARG UNAME=user
RUN useradd -Ums /bin/bash -u $UID $UNAME
USER $UNAME
WORKDIR /home/$UNAME

#RUN alias python=$VENV/bin/python3
#RUN alias pip=$VENV/bin/pip
ENTRYPOINT ["tail", "-f", "/dev/null"]

###############################################################################################
# runtime stage
###############################################################################################
#FROM development AS runtime

#COPY . $PROJDIR
#WORKDIR $PROJDIR
#RUN $VENV/bin/pip install $PROJDIR

#CMD ["./.venv/bin/python", "-m", "$PROJDIR/main.py"]
