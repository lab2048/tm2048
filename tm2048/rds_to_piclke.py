#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 16:03:04 2020

@author: jirlong2
"""

from os import listdir, getcwd
from os.path import isfile, join, isdir

import pyreadr
import pickle
import re

def detectFiletypes(fname, types="txt|tsv|csv"):
    types = types.replace("|", "$|")
    return bool(re.search(types + "$", fname))
    

def rds_to_pickle(fpath = None):
    if fpath == None:
        fpath = getcwd()
        print("Without specify path, set to current directory automatically!")
        print(getcwd())

    if isdir(fpath):
        fns = [f for f in listdir(fpath) if isfile(join(fpath, f)) and detectFiletypes(f.lower(), "rds")]
        for fn in fns:
            result = pyreadr.read_r(fpath + "/" + fn)
            df = result[None]
            fn_p = fn.split(".")[0] + ".p"
            pickle.dump(df, open(fpath + "/" + fn_p, "wb"))





if __name__ == '__main__':
    rds_to_pickle()
