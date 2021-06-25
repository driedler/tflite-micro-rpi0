#!/usr/bin/env bash

set -e


sudo apt-get install -y python3.7-dev python3-pip
python3.7 -m pip install numpy==1.19 pybind11