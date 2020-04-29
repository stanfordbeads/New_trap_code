import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import cv2
import scipy
from scipy import signal
import h5py
import time
import datetime as dt
from tqdm import tqdm
import iminuit
from iminuit import Minuit, describe
from pprint import pprint # we use this to pretty print some stuff later
import glob
import sys
sys.path.append('/home/analysis_user/New_trap_code/Tools/')
import BeadDataFile
from discharge_tools import *
from AnaUtil import *
from joblib import Parallel, delayed
import multiprocessing


def load_dir_reduced_to_time(dirname,file_prefix,max_files):
    '''
    Load time information from the h5 files in a loop into a list. Step size is fixed to 100. 
    '''   
    ## Load all filenames in directory
    var_list = []
    files = []
    [files.append(file_) for file_ in os.listdir(dirname) if file_.startswith(file_prefix) if file_.endswith('.h5')]
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))        
    step_size = 100
    for j in tqdm(np.arange(0,max_files,step_size)):
        BDFs = [BDF.BeadDataFile(dirname+filename) for filename in files[j:j+step_size]]
        [var_list.append(BDFs[k].time[0]/1e9) for k in range(len(BDFs))]
        #[var_list.append(dt.datetime.fromtimestamp(BDFs[k].time[0]/1e9)) for k in range(len(BDFs))]
    return var_list

def load_dir_reduced_to_attr_pos(dirname,file_prefix,max_files):
    '''
    Load time information from the h5 files in a loop into a list. Step size is fixed to 100. 
    '''   
    ## Load all filenames in directory
    var_list = []
    files = []
    [files.append(file_) for file_ in os.listdir(dirname) if file_.startswith(file_prefix) if file_.endswith('.h5')]
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))        
    step_size = 100
    for j in tqdm(np.arange(0,max_files,step_size)):
        BDFs = [BDF.BeadDataFile(dirname+filename) for filename in files[j:j+step_size]]
        [var_list.append(np.mean(BDFs[k].cant_pos[2])) for k in range(len(BDFs))]
        #[var_list.append(dt.datetime.fromtimestamp(BDFs[k].time[0]/1e9)) for k in range(len(BDFs))]
    return var_list


def load_dir_reduced_to_heights(dirname,file_prefix,max_files):
    '''
    Load height information from the h5 files in a loop into a list. Step size is fixed to 100. 
    '''   
    ## Load all filenames in directory
    var_list = []
    files = []
    [files.append(file_) for file_ in os.listdir(dirname) if file_.startswith(file_prefix) if file_.endswith('.h5')]
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))        
    step_size = 100
    for j in tqdm(np.arange(0,max_files,step_size)):
        BDFs = [BDF.BeadDataFile(dirname+filename) for filename in files[j:j+step_size]]
        [var_list.append(BDFs[k].bead_height) for k in range(len(BDFs))]
    return var_list

def environment_processor(year,month,day,bead_date="20200320",bead_number=1,no_bead=False):
    '''
    For a given year,month,day this will process the environmental data (AirTemperature, SurfaceTemperature and Pressure) and save it in the folder of the respective bead (for NoBead data set no_bead to True) as a pickle file using pandas dataframes.
    '''
    date = str(year)+str(month).zfill(2)+str(day).zfill(2) # create the dataname for the file
    
    if(no_bead==False):
        # Create folders for new data. Sometimes a permission problem occurs, solvable by chmod 777 from root.
    
        basic_folder = "/data/new_trap_processed/processed_files/%s/Bead%s/EnvData/" %(bead_date,bead_number)
        output_folder =  basic_folder + "%s" %(date)
        try: os.makedirs(basic_folder)
        except: print("Did not create %s. It may exist or you do not have perimissions." %basic_folder)
        try: os.mkdir(output_folder)
        except: print("Did not create output folder:%s. It may exist or you do not have perimissions."% output_folder)
    
    if(no_bead==True):
        # in case you do not have a bead, dump it in the NoBead folder
        
        no_bead_folder =  "/data/new_trap_processed/processed_files/NoBead/%s" %(date)
        try: os.mkdir(no_bead_folder)
        except: print("Did not create bead folder. It may exist or you do not have perimissions.")
        output_folder =  no_bead_folder
        
    df = pd.DataFrame() # initialize the dataframe to be filled
    time_list, airtemperature_list, surfacetemperature_list, pressure_list, f=([] for i in range(5)) # create lists for any variable stored as column
    for i in np.arange(0,24):# loop through an entire day
        fname = "/data/SC_data/TemperatureAndPressure%s/TempAndPressure%s_%s.hdf5" %(date,date,str(i).zfill(2)) # get the filename 
        try:
            f.append(h5py.File(fname, mode='r+')) # open the file and add it to a file list
        except:
            print("%s hour at %s is not on record" %(i,date)) # if the file does not exist, tell us
            continue       
    for i in np.arange(0,len(f),1): # loop through all files and read out the values and fill them into the respective lists
        try: 
            None
            airtemperature_list.extend(list(f[i]["AirTemperature/AirTemperatures"]))
            surfacetemperature_list.extend(list(f[i]["SurfaceTemperature/SurfaceTemperatures"]))
            pressure_list.extend(list(f[i]["Pressure/Pressures"]))
            time_list.extend(list(f[i]["AirTemperature/Times"]))
        except:
            continue
                
    [f_.close() for f_ in f] # close files    
    # fill the columns of the pandas dataframe
    df["AirTemperature"] = airtemperature_list
    df["SurfaceTemperature"] = surfacetemperature_list
    df["Pressure"] = pressure_list
    df["Time"] = [elements.decode('utf-8') for elements in time_list] # it is saved as a bytes object. Has to be transformed therefore
    df["Time_Human"]= df["Time"].apply(lambda element: "-".join([str(year),str(month).zfill(2),str(day).zfill(2),element])) # Get a date format readable
    df["Time_Epoch"]= df["Time_Human"].apply(lambda element: dt.datetime.timestamp(dt.datetime.strptime(element,"%Y-%m-%d-%H:%M:%S"))) # in epoch
    processed_file_name = output_folder + "/environmental_data_%s%s%s.pkl" %(year,str(month).zfill(2),str(day).zfill(2)) # create the filename
    if(os.path.isfile(processed_file_name)==False):
        #df.to_csv(processed_file_name.replace(".pkl",".csv"),index=False) ## comment back in if you want a .csv for whatever reason
        df.to_pickle(processed_file_name) 
    else: print("Environmental file %s exists already" %processed_file_name)    
    return df

def run_environment_processor_batch(year,month,day_start,day_end,bead_date,bead_number=1):
    """
    Wrapper to run the environmental processor in a practical way. This way it works for an entire month if necessary. We had no bead living longer than a month. So further scripting will not be done until they start to behave as annual events!
    Inputs are: 
    year = year of interest as int, i.e. where the measurement was performed 
    month = same as above
    start_day = the day to start with
    end_day = the day to end with
    bead_date = day the bead was trapped (YYYYMMDD)
    bead_number= number of the bead that date
    """
    day = range(day_start,day_end+1,1)
    year = [year]*len(day)
    month = [month]*len(day)
    dataset_list=(year,month,day)
    dataset_list_T=np.transpose(dataset_list)
    for year,month,day in tqdm(dataset_list_T):
        environment_processor(year,month,day,str(bead_date),bead_number)
    return print("Done")

def match_environmental_data(df,fn):
    '''
    Given a dataframe (df) of heights, one gets the matching environmental data of a list of filenames (fn). If timestamps are to far off a warning will be triggered 
    '''
    df_temp_list = []
    df_temp_list.extend([pd.read_pickle(f) for f in fn]) # read the files in
    df_temp = pd.concat([df_tempi for df_tempi in df_temp_list]).reset_index() # merge the dataframes, you can merge as much as you want since later the time stamps are anyway used. But maybe stay efficient, will ya?
    airt,surf,pressure,values = [[] for i in range(4)]
    for elements in tqdm(range(len(df))): # fun fact: faster than all the possible apply methods. Tried a lot but this is best. DO NOT PARLLELIZE with joblib
        values = df_temp.iloc[(df_temp['Time_Epoch']-df["Time_Epoch"][elements]).abs().argsort()[:1]][['AirTemperature', 'SurfaceTemperature','Pressure','Time_Epoch']].values[0]   # this one subtracts the time_epoch of the data set from the entire column and gets the minimum. Afterwards it adds it if the time difference is less than 10 seconds.
        if(np.abs(values[3]-df["Time_Epoch"][elements])<10):
            airt.append(values[0])
            surf.append(values[1])
            pressure.append(values[2])
        else:print("Time difference is to big!")
    df["AirTemperature"] = airt
    df["SurfaceTemperature"] = surf
    df["Pressure"] = pressure
    return df # get your dataframe back with environmental data
