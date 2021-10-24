import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
import scipy.signal as signal
import scipy.optimize as optimize

import os, fnmatch
import sys, time
from tqdm import tqdm
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
    BDFs = [BDF.BeadDataFile(dirname+filename) for filename in tqdm(files[start_file:start_file+max_file])]

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
        correct_fac = 2.0*n_corr/(n_corr-i) 
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


def digital_demod(dat, fDemod, fs=5000, tFFT=10, win=('tukey',0.25),
        nOverlap=0, detrend='constant', median=False, band_width=2):
    '''
    Single frequency digital demodulation.

    Parameters:
    -----------
    dat: array_like
         Data to demodulate. May contain multiple time series
    fDemod: float
         Frequency at which to do the demodulation.
    fs: float, optional
        Sampling frequency of the time series. Defaults to 2**14=16384 Hz.
    tFFT: float, optional
        Segment length (seconds) to evaluate the FFT. Defaults to 5 s.
    win: tuple, optional
        Input to scipy.signal window function. Defaults to Tukey window with alpha=0.25
    nOverlap: int, optional
        Number of samples to overlap window. Defaults to 0.
    detrend: string, optional
        Input to scipy.signal detrend function. Defaults to 'constant'
    median: Bool, optional
        Median averaging of final result. Defaults to False.
    band_width: bandwidth for the bandpass. Defaults to 2 Hz.

    Returns:
    --------
    result: complex
        Result of the digital demodulation.
    TODO: error handling...
'''
    #band_width=2
    b, a = signal.butter(3, [2.*(fDemod-band_width/2.)/fs, \
                         2.*(fDemod+band_width/2.)/fs ], btype = 'bandpass')
    dat = signal.filtfilt(b, a, dat)
    dat = np.asarray(dat)
    if dat.size==0:
        return(np.empty(dat.shape[-1]))
    nperseg = int(np.round(tFFT*fs)) # Number of segments in a sample segment.
    nOverlap = int(nOverlap);
    # Make the LO time series
    tt = np.arange(len(dat))/fs
    LO = np.exp(-1j*2*np.pi*fDemod*tt)

    # Compute the step to take as we stride through the segments
    step = nperseg - nOverlap
    segShape = ((dat.shape[-1]-nOverlap)//step, nperseg)
    datStrides = (step*dat.strides[-1], dat.strides[-1])
    LOStrides = (step*LO.strides[-1], LO.strides[-1])
    dS = np.lib.stride_tricks.as_strided(dat, shape=segShape, strides=datStrides)
    LOS = np.lib.stride_tricks.as_strided(LO, shape=segShape, strides=LOStrides)

    # Detrend the data
    data = signal.detrend(dS, type=detrend)

    # Demodulate the (windowed) data
    wind = signal.get_window(win,nperseg)
    result = data * wind * LOS
    result = result.sum(axis=-1)
    scale = 2*np.sqrt(1/(wind.sum()**2))
    result *= scale
    return result

########################################################################
##########Neutrality of matter analysis #####################
########################################################################


# extract the response
# this is the same file as the get_scale but with a few more inputs
def get_response(index,folder,drive_freq=71,axis="x",phaseCalib1 = -0.0563,method="SineFit"):
    '''
    index:file
    folder:folder
    drive_freq: frequency you want to analyze, usually f and 2f of your drive
    axis: "x","y" or "z"
    phaseCalib: Offset of phase calibration as extracted from discharge, single electrode or TREK data
    '''
    fname = folder + 'Discharge_'+str(index)+'.h5'
    if(index==1):print(fname)
    neutralityFile = BDF.BeadDataFile(fname=fname)
    bandwidth = 2
    fsamp =5000
    
    if(axis=="x"):
        inSignal=neutralityFile.x2
    if(axis=="y"):
        inSignal=neutralityFile.y2
    if(axis=="z"):
        inSignal=neutralityFile.z2
        
    if(method=="SineFit"):
        ll = likelihood_analyser.LikelihoodAnalyser()
        decimate = 10
        drive_freq1=drive_freq
        fit_kwargs = {'A': 0, 'f': drive_freq1, 'phi': phaseCalib1, 
                      'error_A': 1, 'error_f': 1, 'error_phi': 0.5, 'errordef': 1,
                      'limit_phi': [-2 * np.pi, 2 * np.pi], 
                      'limit_A': [-1, 1], 
                      'print_level': 0, 'fix_f': True, 'fix_phi': True}

        b, a = signal.butter(3, [2.*(drive_freq1-bandwidth/2.)/fsamp, 2.*(drive_freq1+bandwidth/2.)/fsamp ], btype = 'bandpass')
        xx2 = signal.filtfilt(b, a, inSignal)[::decimate]

        m1_tmp = ll.find_mle_sin(xx2, fsamp=5000/decimate, noise_rms=1, plot=False, suppress_print=True, **fit_kwargs)
        response = m1_tmp.values[0]
        
    if(method=="DigiDemod"):
        response = digital_demod(inSignal, fDemod=drive_freq, fs=fsamp, tFFT=10, win=('tukey',0.25),nOverlap=0, detrend='constant', median=False, band_width=bandwidth).imag[0]
    
    return response

### extract voltage from the files on both electrodes
def get_voltage(index,folder):
    trekConvFactor = 200 
    fname = folder + 'Discharge_'+ str(index)+'.h5'
    neutralityFile = BDF.BeadDataFile(fname=fname)
    voltage0= np.std(neutralityFile.electrode_data[0])*np.sqrt(2)*trekConvFactor
    voltage1= np.std(neutralityFile.electrode_data[1])*np.sqrt(2)*trekConvFactor
    return voltage0,voltage1

# wrapper to extract the total voltage applied, 
# the voltage at each electrode and the response at the drive frequency and its 2nd harmonic
def get_response_total(index,folder,drive_freq=71,axis="x",phaseCalib1 = -0.0563,method="SineFit"):
    voltage0,voltage1= get_voltage(index,folder)
    firstVal = get_response(index,folder,drive_freq,axis,phaseCalib1,method)
    secondVal = get_response(index,folder,2*drive_freq,axis,phaseCalib1+np.pi/2,method)
    appliedVoltage=np.abs(np.subtract(voltage0,voltage1))
    return appliedVoltage,firstVal,secondVal,voltage0,voltage1

# the main analysis wrapper
def compare_millicharge_full_analysis_perFile(folder,fileNo=2000,drive_freq=71,axis="x",gap=1,ElectrodeRatio=0.85,
                                              scaleFactor=0,scaleForceFactor=0,
                                              dischargeVoltage=20,method="SineFit",scaleByForce=False):
    
    
    '''
    folder: input folder
    fileNo: number of files
    gap: 0 for both electrodes at the same time, 1 if there is alternating electrode, 2+ if there is a 1+ file break between two alternating electrodes
    ElectrodeRatio: extract ratio from single electrode response
    drive_freq: frequency to be analyzed (at its 2f)
    scaleFactor: Bits to Charge
    scaleForceFactor: Bits to Force
    dischargeVoltage: voltage at which discharge happened
    axis: "x","y" or "z"
    prinValues: debugging
    '''
    
    
    df = pd.DataFrame() # create a data frame
    
    # get an response array including the voltages, run a parallel loop
    
    respArr = np.array(Parallel(n_jobs=32)(delayed(get_response_total)(index=j,folder=folder,drive_freq=drive_freq,axis=axis,method=method) for j in tqdm(range(fileNo))))

    appliedVoltage =respArr.transpose()[0] # from the electrode difference
     
    F=respArr.transpose()[1] # get the response at the drive frequency 
    
    driveDataCharge = np.divide(F,appliedVoltage)*dischargeVoltage/scaleFactor # scale in units of charge
    driveDataForce = 0 # scale in units of force

    G=respArr.transpose()[2] # get the response at 2f
     
    secondDataCharge = np.divide(G,appliedVoltage)*dischargeVoltage/scaleFactor # scale in units of charge
    secondDataForce= 0 # scale in units of force
    
    totalVoltage=[respArr.transpose()[3],respArr.transpose()[4]] # an array to save voltage 
    
    # select whether to scale by charge or force
    if(scaleByForce==True):
        data0=driveDataForce
        data1=secondDataForce
    elif(scaleByForce==False):
        data0=driveDataCharge
        data1=secondDataCharge    
        
    if(gap>0):
        '''
        If the electrodes are alternating there is at least a "gap" of 1 file until the electrode is driven again.
        If there is an additional relaxation gap this gap increases by a factor of 2, e.g. for a single 10s break between
        two alternating electrodes we end up having every 4th file driven at the respective electrode
        '''
        F1=data0[::2*gap] # first electrode
        
        # files without electrode driven or F3=F2 for gap=1. If the gap is larger not all relaxation files will be sampled
        # this is for crosschecks only
        F3=data0[1::2*gap] 
        
        
        F2=data0[gap::2*gap] # second electrode

        G1=data1[::2*gap] # as above for 2f
        G3=data1[1::2*gap] # as above for 2f
        G2=data1[gap::2*gap] # as above for2       

        
        # fill the data frame
        df["F1"] = F1 
        df["F2"] = F2
        df["F3"] = F3

        df["G1"] = G1
        df["G2"] = G2
        df["G3"] = G3
        
        df["A"] = np.add(np.multiply(F2,ElectrodeRatio),F1) #toDO, should this be /(2 x applied voltage * eta1)
        df["A/Um"]= np.multiply(8e-3,np.divide(np.add(np.multiply(F2,ElectrodeRatio),F1),np.mean(appliedVoltage)))
        df["B"] = np.subtract(G1,np.multiply(-G2,ElectrodeRatio*2)) #toDo 

   
    # if both electrodes are driven, no need to slice the data in any way, just extract the f and 2f response
    else: 

            df["F1"] = data0
            df["G1"] = data1
            
    return df,totalVoltage


def quickAnalyzeDataFrame(df):

    
    meanDF = np.mean(df)
    stdDF = np.std(df)/np.sqrt(len(df))
    print(meanDF,stdDF,sep="\n")
    print(np.divide(meanDF,stdDF))

    
    plt.plot(df.F1.ewm(span=10,adjust=True).mean())
    plt.plot(df.G1.ewm(span=10,adjust=True).mean())
    #plt.plot(df.A.ewm(span=10,adjust=True).mean())
    plt.show()

    return meanDF,stdDF

