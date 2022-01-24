#!/usr/bin/env bash

set -e

sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get install -y python3-distutils
sudo apt-get install -y python3.9-dev python3-pip
python3.9 -m pip install numpy~=1.22 pybind11~=2.9