FROM python:3.10 AS builder

# DECLARE SHELL
SHELL ["/bin/bash", "-c"]

## ARGS ##
ARG REQS=base
ARG DEVREQS=test
ARG VENV=/usr/local/gval_env
ARG PROJDIR=/home/user/gval
ARG VERSION='0.0.1'
ARG MAINTANER='Fernando Aristizabal & Gregory Petrochenkov'
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
    $VENV/bin/pip install --no-cache-dir --upgrade build && \
    $VENV/bin/pip install --no-cache-dir -r /tmp/$REQS.txt && \
    $VENV/bin/pip install --no-cache-dir  -r /tmp/$DEVREQS.txt && \
    rm -rf /tmp/*

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
ARG MAINTANER='Fernando Aristizabal & Gregory Petrochenkov'
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
# add path to virtual env so that future python commands use it
ENV PATH="$VENV/bin:$PATH"

## ADDING USER GROUP ##
ARG UID=1001
ARG UNAME=user
RUN useradd -Ums /bin/bash -u $UID $UNAME
USER $UNAME
WORKDIR /home/$UNAME

# RETRIEVE BUILT DEPENDENCIES
COPY --from=builder --chown=$UID $VENV $VENV

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
