""" Utility functions used by the tool """
from hashlib import md5
import pandas as pd


def hash_file(path):
    """ Returns md5 hash of a file"""
    chk = md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            chk.update(chunk)
    return chk.hexdigest()


def return_cols(path):
    """ Returns column names of the CSV"""
    data = pd.read_csv(path)
    return list(data.columns)


def select_cols(path, cols):
    """Select passed columns from the CSV"""
    data = pd.read_csv(path)
    return data[data.columns.intersection(cols)]
