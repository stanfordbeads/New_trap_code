import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as signal
import scipy.optimize as optimize

import os, fnmatch
import sys, time

import BeadDataFile as BDF

import sys
sys.path.append('/home/analysis_user/New_trap_code/Tools/StatFramework/')
from likelihood_calculator import likelihood_analyser

def load_dir(dirname, file_prefix = 'Discharge', start_file=0, max_file=500):
    ''' Load all files in directory to a list of BeadDataFile
        INPUTS: dirname, directory name
        max_file, maximum number of files to read'''

    ## Load all filenames in directory
    files = []
    [files.append(file_) for file_ in os.listdir(dirname) if file_.startswith(file_prefix) if file_.endswith('.h5')]
    files.sort(key=lambda ff: int(os.path.splitext(ff)[0].split('_')[-1])) ## sort by index

    # Load data into a BeadDataFile list
    BDFs = [BDF.BeadDataFile(dirname+filename) for filename in files[start_file:start_file+max_file]]

    print(len(files),' files in folder')
    print(len(BDFs),' files loaded')

    return BDFs

def load_dir_sorted(dirname, file_prefix = 'Discharge',start_file=0, max_file=500):
    ''' Load all files in directory to a list of BeadDataFile
        INPUTS: dirname, directory name
        max_file, maximum number of files to read'''
        
    ## Load all filenames in directory
    files = []
    [files.append(file_) for file_ in os.listdir(dirname) if file_.startswith(file_prefix) if file_.endswith('.h5')]
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))    
    # Load data into a BeadDataFile list
    BDFs = [BDF.BeadDataFile(dirname+filename) for filename in files[start_file:start_file+max_file]]
    print(len(files),' files in folder')
    print(len(BDFs),' files loaded')
    return BDFs



def discharge_response(foldername, str_axis, drive_freq,max_file=500):
    bdfs = load_dir(foldername,max_file=max_file)
    resp = [np.std(bb.response_at_freq2(str_axis, drive_freq=drive_freq)) for bb in bdfs]
    return resp


########################################################################
########################################################################
########################################################################



def correlation(drive, response, fsamp, fdrive, filt = False, band_width = 1):
    '''Compute the full correlation between drive and response,
       correctly normalized for use in step-calibration.

       INPUTS:   drive, drive signal as a function of time
                 response, resposne signal as a function of time
                 fsamp, sampling frequency
                 fdrive, predetermined drive frequency
                 filt, boolean switch for bandpass filtering
                 band_width, bandwidth in [Hz] of filter

       OUTPUTS:  corr_full, full and correctly normalized correlation'''

    ### First subtract of mean of signals to avoid correlating dc
    drive = drive-np.mean(drive)
    response = response-np.mean(response)

    ### bandpass filter around drive frequency if desired.
    if filt:
        b, a = signal.butter(3, [2.*(fdrive-band_width/2.)/fsamp, \
                             2.*(fdrive+band_width/2.)/fsamp ], btype = 'bandpass')
        drive = signal.filtfilt(b, a, drive)
        response = signal.filtfilt(b, a, response)
    
    ### Compute the number of points and drive amplitude to normalize correlation
    lentrace = len(drive)
    drive_amp = np.sqrt(2)*np.std(drive)

    ### Define the correlation vector which will be populated later
    corr = np.zeros(int(fsamp/fdrive))

    ### Zero-pad the response
    response = np.append(response, np.zeros(int(fsamp / fdrive) - 1) )

    ### Build the correlation
    n_corr = len(drive)
    for i in range(len(corr)):
        ### Correct for loss of points at end
        correct_fac = 2.0*n_corr/(n_corr-i) ### x2 from empirical test
        corr[i] = np.sum(drive*response[i:i+n_corr])*correct_fac

    return corr * (1.0 / (lentrace * drive_amp))

def build_z_response_discharge(bdf_list, drive_freq, bandwidth=1, decimate=10, phase=0.2, fix_phase=False):
    la = likelihood_analyser.LikelihoodAnalyser()
    fit_kwargs = {'A': 0, 'f': drive_freq, 'phi': phase,
        'error_A': 1, 'error_f': 1, 'error_phi': 0.5, 'errordef': 1,
        'limit_phi': [-np.pi, np.pi],
        'limit_A': [-10000, 10000],
        'print_level': 0, 'fix_f': True, 'fix_phi': fix_phase}
    
    m1_list = []
    for bb in bdf_list:                        
        frequency = fit_kwargs['f']
        xx2 = bb.response_at_freq2('z', frequency, bandwidth=bandwidth)
        xx2 = xx2[5000:-5000:decimate]  # cut out the first and last second

        m1_tmp = la.find_mle_sin(xx2, fsamp = 5000 / decimate, noise_rms=1, **fit_kwargs)
        m1_list.append(m1_tmp)
    phases = np.array([m1_.values[2] for m1_ in m1_list])
    amps = np.array([m1_.values[0] for m1_ in m1_list])
                                                                            
    return amps, phases

