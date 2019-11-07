import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as signal
import scipy.optimize as optimize

import os, fnmatch
import sys, time

import BeadDataFile as BDF


def load_dir(dirname, file_prefix = 'Discharge', max_file=500):
    ''' Load all files in directory to a list of BeadDataFile
        INPUTS: dirname, directory name
        max_file, maximum number of files to read'''
        
    ## Load all filenames in directory
    files = []
    [files.append(file_) for file_ in os.listdir(dirname) if file_.startswith(file_prefix) if file_.endswith('.h5')]
    files.sort(key=lambda ff: int(os.path.splitext(ff)[0].split('_')[-1])) ## sort by index
    
    # Load data into a BeadDataFile list
    BDFs = [BDF.BeadDataFile(dirname+filename) for filename in files[:max_file]]
    
    print(len(files),' files in folder')
    print(len(BDFs),' files loaded')
    
    return BDFs


def discharge_response(foldername, str_axis, drive_freq):
    
    bdfs = load_dir(foldername)
    resp = [np.std(bb.response_at_freq2(str_axis, drive_freq=drive_freq)) for bb in bdfs]

    return resp
