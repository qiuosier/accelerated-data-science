#!/bin/bash

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


### Usage: sh scripts/runtox.sh py3.8 <pytest-args>
### Runs all environments with substring py3.8 and the given arguments for pytest

set -ex

if [ -n "$TOXPATH" ]; then
    true
elif which tox &> /dev/null; then
    TOXPATH=tox
else
    TOXPATH=./.venv/bin/tox
fi

searchstring="$1"

export TOX_PARALLEL_NO_SPINNER=1
exec $TOXPATH -p auto -e "$($TOXPATH -l | grep "$searchstring" | tr $'\n' ',')" -- "${@:2}"
