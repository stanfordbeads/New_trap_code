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
from analysis_tools import *
from height_tools import *
from joblib import Parallel, delayed
import multiprocessing


def load_dir_reduced_to_qpd_sum(dirname,file_prefix,max_files):
    '''
    Load time information from the h5 files in a loop into a list. Step size is fixed to 100. 
    '''   
    ## Load all filenames in directory
    var_list = []
    files = []
    [files.append(file_) for file_ in os.listdir(dirname) if file_.startswith(file_prefix) if file_.endswith('.h5')]
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    print(len(files))
    step_size = 50
    for j in tqdm(np.arange(0,max_files,step_size)):
        BDFs = [BDF.BeadDataFile(dirname+filename) for filename in files[j:j+step_size]]
        [var_list.append(BDFs[k].quad_sum) for k in range(len(BDFs))]
        #[var_list.append(dt.datetime.fromtimestamp(BDFs[k].time[0]/1e9)) for k in range(len(BDFs))]
    return var_list

def load_dir_reduced_to_spin(dirname,file_prefix,max_files):
    '''
    Load time information from the h5 files in a loop into a list. Step size is fixed to 100. 
    '''   
    ## Load all filenames in directory
    var_list = []
    files = []
    [files.append(file_) for file_ in os.listdir(dirname) if file_.startswith(file_prefix) if file_.endswith('.h5')]
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    print(len(files))
    step_size = 50
    for j in tqdm(np.arange(0,max_files,step_size)):
        BDFs = [BDF.BeadDataFile(dirname+filename) for filename in files[j:j+step_size]]
        [var_list.append(BDFs[k].spin_data) for k in range(len(BDFs))]
        #[var_list.append(dt.datetime.fromtimestamp(BDFs[k].time[0]/1e9)) for k in range(len(BDFs))]
    return var_list


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


def load_dir_reduced_to_height(dirname,file_prefix,max_files):
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

def spin_processor(bead_date,bead_number,dataset,run,max_files,fsamp):
    dirname ="/data/new_trap/" + str(bead_date) + "/Bead%s/" %bead_number +dataset
    fsamp = fsamp 
    res = fsamp
    spin_time = load_dir_reduced_to_spin(dirname,run,max_files)
    df_spin = pd.DataFrame()
    df_spin["spin_time"] = spin_time
    a_list = []
    p_list = []
    # for loop seem to be as slow as this
    df_spin["spin_amp"] = df_spin.spin_time.apply(lambda element: spin_data_to_amp_and_phase(element,fsamp,res)[1])
    df_spin["spin_phase"] = df_spin.spin_time.apply(lambda element: spin_data_to_amp_and_phase(element,fsamp,res)[2])
    #if(save_file==True):
    #    df.to_pickle(filename)
    return df_spin


def harmonics_processor_input(bead_date,bead_number,dataset,run,start_file=0,max_file=5,no_harmonics=10,res=5000,save_file=True):
    path="/harmonics/"
    dirname ="/data/new_trap/" + str(bead_date) + "/Bead%s/" %bead_number +dataset
    max_input_length = get_max_file_length(dirname, file_prefix = run)
    files = load_dir_sorted(dirname, run,start_file=start_file,  max_file=max_file)
    if(start_file+max_file>max_input_length):
        return print("The file number you specficed exeeds the maximum.")
    fsamp = files[0].fsamp
    shake_freq = int(files[0].cant_freq)
    harmonic_list_x, sideband_list_x, phase_list_x, sidephase_list_x =([] for i in range(4))
    harmonic_list_y, sideband_list_y, phase_list_y, sidephase_list_y =([] for i in range(4))
    harmonic_list_z, sideband_list_z, phase_list_z, sidephase_list_z =([] for i in range(4))
    print(shake_freq)
    xmean_list,ymean_list,zmean_list = ([] for i in range(3))
    cant_xpos_list,cant_ypos_list,cant_zpos_list=([] for i in range(3))      
    freq_list,time_stamp_list = ([] for i in range(2))
    stroke_list = []
    x_feedback_list,y_feedback_list,z_feedback_list = ([] for i in range(3))
    
    for i in np.arange(0,len(files),1):
        #print(files[i].fname)
        data = files[i].xyz2
        
        xmean_list.append(np.mean(files[i].x2))
        ymean_list.append(np.mean(files[i].y2))
        zmean_list.append(np.mean(files[i].z2))
        
        cant_xpos_list.append(np.mean(files[i].cant_pos[0]))
        cant_ypos_list.append(np.mean(files[i].cant_pos[1]))
        cant_zpos_list.append(np.mean(files[i].cant_pos[2]))
        
        stroke_list.append(np.mean(estimate_stroke(files[i].cant_pos[1])))
        
        x_feedback_list.append(np.mean(files[i].feedback[0]))
        y_feedback_list.append(np.mean(files[i].feedback[1]))
        z_feedback_list.append(np.mean(files[i].feedback[2]))

        time_stamp_list.append(files[i].time[0])
        
        FFT_and_phases = data_to_amp_and_phase(data,fsamp,res)
        freqs,harmonics_x,sidebands_x = get_harmonics_with_sideband(FFT_and_phases[0],shake_freq=shake_freq,no_harmonics=no_harmonics)
        _,harmonics_y,sidebands_y = get_harmonics_with_sideband(FFT_and_phases[1],shake_freq=shake_freq,no_harmonics=no_harmonics)
        _,harmonics_z,sidebands_z = get_harmonics_with_sideband(FFT_and_phases[2],shake_freq=shake_freq,no_harmonics=no_harmonics)
        
        freq_list.append(freqs)
        
        harmonic_list_x.append(np.sqrt(harmonics_x))     
        harmonic_list_y.append(np.sqrt(harmonics_y))     
        harmonic_list_z.append(np.sqrt(harmonics_z))     
        
        sideband_list_x.append(np.sqrt(sidebands_x))     
        sideband_list_y.append(np.sqrt(sidebands_y))     
        sideband_list_z.append(np.sqrt(sidebands_z))     

        
        _,phases_x,sidephases_x = get_harmonics_with_sideband(FFT_and_phases[3],shake_freq=shake_freq,no_harmonics=no_harmonics)
        _,phases_y,sidephases_y = get_harmonics_with_sideband(FFT_and_phases[4],shake_freq=shake_freq,no_harmonics=no_harmonics)# should be 4?
        _,phases_z,sidephases_z = get_harmonics_with_sideband(FFT_and_phases[5],shake_freq=shake_freq,no_harmonics=no_harmonics)# should be 5?
        
        phase_list_x.append(phases_x)
        phase_list_y.append(phases_y)
        phase_list_z.append(phases_z)

        sidephase_list_x.append(sidephases_x)
        sidephase_list_y.append(sidephases_y)
        sidephase_list_z.append(sidephases_z)

        
    # make the dataframe and fill it
    df = pd.DataFrame()
    
    df["start_time"]=time_stamp_list
    
    df["stroke"]=stroke_list
    
    df["x_mean"]=xmean_list
    df["y_mean"]=ymean_list
    df["z_mean"]=zmean_list
    
    df["attractor_position_x"]=cant_xpos_list
    df["attractor_position_y"]=cant_ypos_list
    df["attractor_position_z"]=cant_zpos_list
    
    # frequencies
    df["frequency"] = freq_list
    
    df["amplitude_x"] = harmonic_list_x
    df["amplitude_y"] = harmonic_list_y
    df["amplitude_z"] = harmonic_list_z
    
    df["phase_x"] = phase_list_x
    df["phase_y"] = phase_list_y
    df["phase_z"] = phase_list_z
    
    df["sideband_amplitude_x"] = sideband_list_x
    df["sideband_amplitude_y"] = sideband_list_y
    df["sideband_amplitude_z"] = sideband_list_z

    df["sideband_phase_x"] = sidephase_list_x
    df["sideband_phase_y"] = sidephase_list_y
    df["sideband_phase_z"] = sidephase_list_z    
    
    df["x_feedback"] = x_feedback_list
    df["y_feedback"] = y_feedback_list
    df["z_feedback"] = z_feedback_list
    
    if(save_file==True):
        save_processed_file(df,bead_date=bead_date,bead_number=bead_number,dataset=dataset,run=run,process_type="main",create_csv=True)
    return df

def save_processed_file(df,bead_date,bead_number,dataset,run,process_type="main",create_csv=False):
    base_proc = "/data/new_trap_processed/processed_files/" + str(bead_date) +  "/Bead%s/" %bead_number
    try:
        os.makedirs(base_proc+dataset)        
        print("Created subdirs %s" %dataset)
    except: 
        print("Folder exists or you do not have permissions")
    processed_file_name = base_proc + dataset +  run + "_%s.pkl" %process_type 
    if(os.path.isfile(processed_file_name)==False):
        df.to_pickle(processed_file_name)
        if(create_csv==True):
            df.to_csv(processed_file_name.replace(".pkl",".csv"),index=False)
        print(processed_file_name)
    else:
        print("Processed file could not be saved, probably exists.")
    return None

def get_max_file_length(dirname, file_prefix = 'Discharge'):
    ''' Load all files in directory to a list of BeadDataFile
        INPUTS: dirname, directory name
        max_file, maximum number of files to read'''
        
    ## Load all filenames in directory
    files = []
    [files.append(file_) for file_ in os.listdir(dirname) if file_.startswith(file_prefix) if file_.endswith('.h5')]
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))   
    return len(files)


def loop_run_harmonics_processor(bead_date,bead_number,dataset,run,start_file,max_file,no_harmonics=10,res=5000,save_file=True):    
    df_tot = pd.DataFrame()
    start_file = start_file
    end_file = max_file
    step_size=100
    for i in tqdm(np.arange(start_file,end_file,step_size)):
        try:
            df_temp = pd.DataFrame()
            df_temp = harmonics_processor_input(bead_date,bead_number,dataset,run,i,step_size,no_harmonics,res,save_file=False)
            df_tot = pd.concat([df_tot,df_temp],ignore_index=True)
        except: print("The -%d-th file did not work. Maybe your specified max_file is longer than the datasets" %j)        
    if(save_file==True):
        save_processed_file(df_tot,bead_date=bead_date,bead_number=bead_number,dataset=dataset,run=run,process_type="main",create_csv=True)
    return df_tot


def reduced_df(df_in,parameter=["x_mean","y_mean","z_mean","amplitude_x","amplitude_y","amplitude_z","phase_x","phase_y","phase_z"],frequency=3,no_harmonics=15):
    df_red = pd.DataFrame()
    for elements in tqdm(parameter):
        if(df_in["%s" %elements].dtype=="O"):
            for j in range(no_harmonics):
                list_temp = []
                for k in range(len(df_in)):
                    list_temp.append(df_in["%s" %elements][k][j])
                df_red["%s_%d" %(elements,j*frequency+frequency)] = list_temp 
        elif(df_in["%s" %elements].dtype=="float64"):
            df_red["%s" %(elements)] = df_in["%s" %elements]
    df_red=df_red.reset_index()
    return df_red


''' NOT IN USE ANYMORE
def processor_new_trap(folder_basic,distances,frequency_list,start_file=0,max_file=100,no_harmonics=15,res_factor=10,save_file=True):
    df_list=[]
    for i in range(len(distances)):
        distance = distances[i]
        shake_freq = frequency_list[i]
        print("The used distance is %s and the used frequency is %s" %(distance,shake_freq))   
        folder_shaking = "/Shaking/Shaking%d/" %distance
        folder = folder_basic + folder_shaking
        df = harmonics_processor_input(folder,filename_input="Shaking3_",filename_output="Shaking%s_%d" %(distance,i),start_file=start_file,max_file=max_file,shake_freq=shake_freq,no_harmonics=no_harmonics,res_factor=res_factor,save_file=save_file)
        df_list.append(df)
    return df_list
    
    
# THE FILE LOADER FOR PROCESSED FILES

def file_loader_processed(folder_processed,folder_list):
    df_list = []
    for folder_list_entry in np.unique(folder_list):
        folder = folder_processed+"/Shaking%d/" %folder_list_entry
        for output_file_number in np.arange(0,folder_list.count(folder_list_entry),1):
            file_to_load = folder+"harmonics_processed_basic_Shaking%d_%d.pkl" %(folder_list_entry,output_file_number)
            print(file_to_load )
            try:df = pd.read_pickle(file_to_load)
            except:print("Did not load. File %s does not exist" %(file_to_load))
            df_list.append(df)
    return df_list    

'''