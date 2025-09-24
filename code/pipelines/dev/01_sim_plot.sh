#!/bin/bash

# Copyright 2025 Ankur Sinha
# Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
# File :
#


# Simulate and plot the network

if [ $# -ne 1 ]
then
    echo "Takes only one parameter: the model parameters file"
    exit -1
fi

MODEL_PARAMS_FILE="$1"

python GolgiCellNetwork.py --model-parameters-file "$MODEL_PARAMS_FILE" create-model-simulation-run && pynml-plottimeseries -offset LEMS*xml
