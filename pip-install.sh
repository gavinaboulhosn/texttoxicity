#!/bin/bash

function pip_install {
    pip install $1 && pip freeze | grep $1 >> requirements.txt;
}