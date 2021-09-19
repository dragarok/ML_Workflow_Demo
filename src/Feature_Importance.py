# -*- coding: utf-8 -*-
# Alok's python Feature Importance Script here

import yaml


def read_config(fname="src/config.yaml"):
    """ Function to read and return config params from yaml file

    Args:
        fname (str) : File name for yaml file
    Returns:
        dict : Dictionary with parameters from yaml file

    """
    with open(fname, "r") as fs:
        try:
            return yaml.safe_load(fs)
        except yaml.YAMLError as exc:
            print(exc)
            return

