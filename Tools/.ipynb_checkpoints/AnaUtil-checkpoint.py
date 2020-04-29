# import stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
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


# basic functions and cost functions

# for use the example notebook in New_Trap_Code/Scripts

def gaussian(x,params=list):
    '''
    Generic defintion of a Gaussian
    '''
    norm = (1/((1/2*params[2])*np.sqrt(np.pi * 2)))
    return params[0] * norm * np.exp(-(np.subtract(x,params[1])**2/(2*params[2]**2)))+params[3]

def linear(x,params=list):
    return params[0]*x+params[1]

def chisquare_1d(function, functionparams, data_x, data_y,data_y_error):
    chisquarevalue=np.sum(np.power(np.divide(np.subtract(function(data_x,functionparams),data_y),data_y_error),2))
    ndf = len(data_y)-len(functionparams)
    #print(ndf)
    return (chisquarevalue, ndf)

def chisquare_gaussian(area,mean,sigma,constant):
    return chisquare_1d(function=gaussian,functionparams=[area,mean,sigma,constant],data_x=data_x,data_y=data_y,data_y_error=data_y_error)[0]

def chisquare_linear(a,b):
    return chisquare_1d(function=linear,functionparams=[a,b],data_x=data_x,data_y=data_y,data_y_error=data_y_error)[0]


# calibration of the voltage - position conversion

def voltage_to_x_position(voltage,slope=0.019834000085488412,offset=-0.0015000315197539749,redo=False):
    if(redo==True):
        
        pos_list=np.asarray([-0.007,4.968,9.91])
        y_err=np.asarray([0.01,0.01,0.01])
        val = np.asarray([0,250,500])
        data_x=val
        data_y=pos_list
        data_y_error=y_err
        m2=Minuit(chisquare_linear, 
             a = 100,
             b =0,
             errordef = 1,
             print_level=1)
        m2.migrad()
        print(m2.values["a"],m2.values["b"])
        slope = m2.values["a"]
        offset = m2.values["b"]
        plt.plot(val,pos_list,marker="*")
        plt.plot(val,m2.values["a"]*val+m2.values["b"])
    position=(voltage-offset)/slope
    return poosition
    return position

# calibration of the voltage - position conversion


def voltage_to_z_position(voltage,slope=0.1,offset=0.0,redo=False):
    if(redo==True):
        def chisquare_linear(a,b):
            return chisquare_1d(function=linear,functionparams=[a,b],data_x=data_x,data_y=data_y,data_y_error=data_y_error)[0]

        pos_list=np.asarray([-0.005,9.90])
        y_err=np.asarray([0.01,0.01])
        val = np.asarray([0,100])
        data_x=val
        data_y=pos_list
        data_y_error=y_err
        m2=Minuit(chisquare_linear, 
             a = 100,
             b =0,
             errordef = 1,
             print_level=1)
        m2.migrad()
        print(m2.values["a"],m2.values["b"])
        slope = m2.values["a"]
        offset = m2.values["b"]
        plt.plot(val,pos_list,marker="*")
        plt.plot(val,m2.values["a"]*val+m2.values["b"])
    position=(voltage-offset)/slope
    return position
# helper functions

## extract the stroke (toDO add frequency)
def extract_freq_and_stroke(cant_pos_y):
    stroke = voltage_to_x_position(np.std(cant_pos_y)*np.sqrt(2))
    return 2*stroke


# extract the harmonics and a sideband for a given data set with a given frequency, also able to pick the side bands properly
# toDO: proper averaging for signals and sidebands

def get_harmonics_with_sideband(input_psd,shake_freq,no_harmonics,res_factor=10,sum_range=0,sideband_spacing=-7,plot=False):
    harmonics = []
    sideband = []
    FreqTF = np.arange(shake_freq,(shake_freq*no_harmonics+shake_freq),shake_freq)
    for i in range(no_harmonics):
        harmonics_sum = input_psd[shake_freq*res_factor*(i+1)]+ input_psd[(shake_freq+sum_range)*res_factor*(i+1)]+input_psd[(shake_freq-sum_range)*res_factor*(i+1)]
        harmonics.append(harmonics_sum)
        #side_up = input_psd[(shake_freq+sideband_spacing)*res_factor*(i+1)]
        #side_down = input_psd[(shake_freq-sideband_spacing)*res_factor*(i+1)]
        #sideband.append(np.divide(np.sum(side_up,side_down),2))
        sideband.append(input_psd[(shake_freq+sideband_spacing)*res_factor*(i+1)])
    if(plot==True):
        plt.plot(FreqTF,harmonics, marker ="o", linestyle ="")
        plt.yscale("log")
        #plt.show()
    #print(harmonics_sum)
    return FreqTF,harmonics,sideband       

# obtain amplitude and phase of a complex psd
def data_to_amp_and_phase(data,fsamp,res):
    data_det=signal.detrend(data)
    res = res
    fsamp = fsamp # stays hard coded for now
    freqs=np.linspace(0,fsamp/2,(res/2)+1)    # change
    xFFT=np.fft.rfft(data_det[0])
    yFFT=np.fft.rfft(data_det[1])
    zFFT=np.fft.rfft(data_det[2])
    norm = np.sqrt(2 / (res* fsamp))
    xpsd = norm**2 * (xFFT * xFFT.conj()).real
    ypsd = norm**2 * (yFFT * yFFT.conj()).real
    zpsd = norm**2 * (zFFT * zFFT.conj()).real
    xphase=np.angle(xFFT)
    yphase=np.angle(yFFT)
    zphase=np.angle(zFFT)
    return xpsd,ypsd,zpsd,xphase,yphase,zphase

# hack for spin data - 1d vs 3d in the other one
## toDO: make this one proper function
def spin_data_to_amp_and_phase(data,fsamp,res):
    data_det=signal.detrend(data)
    res = res
    fsamp = fsamp # stays hard coded for now
    freqs=np.linspace(0,fsamp/2,(res/2)+1)    # change
    xFFT=np.fft.rfft(data_det)
    norm = np.sqrt(2 / (res* fsamp))
    xpsd = norm**2 * (xFFT * xFFT.conj()).real
    xphase=np.angle(xFFT)
    return freqs,xpsd,xphase



### PROCESSOR FUNCTION ###
'''
This is the function to process files from the new trap to some quick analysis regarding the harmonics or sidebands. Includes feedback and spin.


toDO: Temperature inclusion

'''

def harmonics_processor_input(folder,filename_input,filename_output,max_file=5,shake_freq=13,no_harmonics=10,fsamp=5000,res=50000,res_factor=10,save_file=True):
    path="/harmonics/"
    #try:
    #    os.mkdir(path)
    #except OSError:
    #    print ("Creation of the directory %s failed" % path)
    #else:
    #    print ("Successfully created the directory %s " % path)
        
    # load files and initialize processor
    files = load_dir_sorted(folder, file_prefix =filename_input, max_file=max_file)

    harmonic_list_x, sideband_list_x, phase_list_x, sidephase_list_x =([] for i in range(4))
    harmonic_list_y, sideband_list_y, phase_list_y, sidephase_list_y =([] for i in range(4))
    harmonic_list_z, sideband_list_z, phase_list_z, sidephase_list_z =([] for i in range(4))

    xmean_list,ymean_list,zmean_list = ([] for i in range(3))
    cant_xpos_list,cant_ypos_list,cant_zpos_list=([] for i in range(3))      
    freq_list,time_stamp_list = ([] for i in range(2))
    stroke_list = []
    spin_list = []
    x_feedback_list,y_feedback_list,z_feedback_list = ([] for i in range(3))
    
    for i in tqdm(np.arange(0,len(files),1)):
        #print(files[i].fname)
        data = files[i].xyz2
        spin_data = files[i].spin_data
    
        
        freq,spin_amp, spin_phase = spin_data_to_amp_and_phase(spin_data,10*fsamp,10*res)
        spin_list.append(spin_amp)
        
        xmean_list.append(np.mean(files[i].x2))
        ymean_list.append(np.mean(files[i].y2))
        zmean_list.append(np.mean(files[i].z2))
        
        cant_xpos_list.append(np.mean(files[i].cant_pos[0]))
        cant_ypos_list.append(np.mean(files[i].cant_pos[1]))
        cant_zpos_list.append(np.mean(files[i].cant_pos[2]))
        
        stroke_list.append(np.mean(extract_freq_and_stroke(files[i].cant_pos[1])))
        
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
        _,phases_y,sidephases_y = get_harmonics_with_sideband(FFT_and_phases[3],shake_freq=shake_freq,no_harmonics=no_harmonics)
        _,phases_z,sidephases_z = get_harmonics_with_sideband(FFT_and_phases[3],shake_freq=shake_freq,no_harmonics=no_harmonics)
        
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
    #print(len(spin_amp))
    df["spin_data"] = spin_list
    
    df["x_feedback"] = x_feedback_list
    df["y_feedback"] = y_feedback_list
    df["z_feedback"] = z_feedback_list
    
    if(save_file==True):
        processed_par_folder = "/data/new_trap_processed/harmonics_processed"
        processed_folder= processed_par_folder+"%s" %(folder[14:30]+folder[38:])
        try: os.mkdir(processed_folder)
        except: print("Folder exists or no permission") 
        processed_file_name = processed_folder + "harmonics_processed_basic_%s.pkl" %filename_output
        if(os.path.isfile(processed_file_name)==False):
            df.to_csv(processed_file_name.replace(".pkl",".csv"),index=False)
            df.to_pickle(processed_file_name)
            print(processed_file_name)
        else:print("File could not be saved, probably exists.")
    return df

# Now use it to actually process all files you need 
# toDO: read frequencies out automatically

def processor_new_trap(folder_basic,distances,frequency_list,max_file=100,no_harmonics=15,res_factor=10,save_file=True):
    df_list=[]
    for i in range(len(distances)):
        distance = distances[i]
        shake_freq = frequency_list[i]
        print("The used distance is %s and the used frequency is %s" %(distance,shake_freq))   
        folder_shaking = "/Shaking/Shaking%d/" %distance
        folder = folder_basic + folder_shaking
        df = harmonics_processor_input(folder,filename_input="Shaking%d_" %i,filename_output="Shaking%s_%d" %(distance,i),max_file=max_file,shake_freq=shake_freq,no_harmonics=no_harmonics,res_factor=res_factor,save_file=save_file)
        df_list.append(df)
    return 
    
    
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



### ANALYSIS FUNCTIONS###
"""
For fast analysis
"""

# obtain mean and sum
# ToDO: MEDIAN AND STD, does not work for now because of the pandas way of doing things
def get_mean_and_sum_of_harmonics(df,axis="x",no_harmonics=10,norm_factor=1):
    mean_list, std_list, sum_list, median_list=([] for i in range(4))
    for i in np.arange(0,no_harmonics,1):
        mean_list.append(list(pd.DataFrame.mean(df["amplitude_%s" %axis]))[i])
        sum_list.append(list(pd.DataFrame.sum(df["amplitude_%s" %axis]))[i])
    return np.divide(mean_list,norm_factor),np.divide(sum_list,norm_factor)



def plot_basic_harmonic_sum(df_list,axis,norm_factor=1,no_harmonics=15,label="label",var_of_interest="attractor_position_z",output_file="test.png",legend=True,save_file=False):
    "returns the harmonic sum for the data frame normed by the length of the data file"
    sum_list =[]
    x_axis = np.arange(1,no_harmonics+1,1)
    for i in range(len(df_list)):
        _,sumws = get_mean_and_sum_of_harmonics(df_list[i],axis=axis,no_harmonics=no_harmonics,norm_factor=norm_factor)
        sum_list.append(sumws)
        plt.plot(x_axis,np.divide(sumws,len(df_list[i])),linestyle="-",marker="o", label="%s: " %label + str(np.mean(df_list[i]["%s" %var_of_interest]))[:5])#np.mean(df_sub_list[i]["attractor_position_z"]))
        plt.yscale("log")
        if(norm_factor==1):plt.ylabel(r" sum NSD in %s [AU/$\sqrt{Hz}$]"% axis)
        if(norm_factor!=1):plt.ylabel(r" saverage NSD in %s [N/$\sqrt{Hz}$]" %axis)
        plt.xlabel("harmonic")
    if(legend==True):plt.legend()
    if(save_file==True): plt.savefig(output_file, dpi=300, bbox_inches="tight")    
    plt.show()
    return sum_list

def plot_amplitude_vs_data_set(df,axis="x",no_harmonics=15,save=False,save_folder=None,save_filename=None,file_type = ".png"):
    time=np.arange(10,len(df[0])*10+10,10)
    total_outputs=[]
    for k in np.arange(0,no_harmonics,1):
        for j in np.arange(0,len(df)):
            output_list=[]
            for i in range(len(df[0])):
                output_list.append(df[j]["amplitude_%s" %axis][i][k])
            plt.plot(time,output_list)
            total_outputs.append(output_list)    
        plt.title("Harmonic: %d"%(k+1))    
        plt.yscale("log")
        plt.xlabel("time [s]")
        plt.ylabel(" %s NSD [a.u.$\sqrt{Hz}$]" %axis)
        #plt.ylim(0.01*np.min(output_list),2*np.max(output_list))
        if(save==True):
            save_as = save_folder + "harmonics_ampl_vs_ds/" + save_filename + "%d"%(k+1) + file_type
            plt.savefig(save_as, dpi=300,bbox_inches="tight")
        plt.show()    
    return time,total_outputs

def plot_harmonics_per_data_set(df,normalized=False,which_harmonic=0,files=10,save=False,save_folder=None,save_filename=None,file_type = ".png"):
    harmonics=np.arange(1,len(df[0]["amplitude_x"][0])+1,1)
    for i in range(files):
            for j in np.arange(0,len(df)):
                if(normalized==True):    
                    plt.plot(harmonics,df[j]["amplitude_x"][i]/df[j]["amplitude_x"][i][which_harmonic])
                else:
                    plt.plot(harmonics,df[j]["amplitude_x"][i],linestyle="-",marker="o")
            plt.title("File: %d"%(i+1))    
            plt.yscale("log")
            plt.xlabel("harmonic ")
            plt.ylabel("NSD [a.u.$\sqrt{Hz}$]")
            if(save==True):
                save_as = save_folder + "harmonics_per_ds/" + save_filename + "%d"%(i+1) + file_type
                plt.savefig(save_as, dpi=300,bbox_inches="tight")
            plt.show()
    return   

def plot_pos_mean_vs_dataset(df,axis = ["x","y","z"],save=False,save_folder=None,save_filename=None,file_type = ".png"):
    time=np.arange(10,len(df[0])*10+10,10)
    for element in axis:
        for i in np.arange(0,len(df),1):
            plt.plot(time,df[i]["%s_mean" %element])
        #plt.legend()
        plt.ylabel(r"%s_pos [AU]" %element)
        plt.xlabel("time [s]")
        #plt.yscale("log")
        #plt.savefig("%s_mean.png" %element, dpi=300, bbox_inches ="tight")
        if(save==True):
            save_as = save_folder + "mean_position/" + save_filename + "%s"%(element) + file_type
            plt.savefig(save_as, dpi=300,bbox_inches="tight")
        plt.show()
        
    return    
def plot_attr_pos_vs_dataset(df,axis = ["x","y","z"],save=False,save_folder=None,save_filename=None,file_type = ".png"):
    time=np.arange(10,len(df[0])*10+10,10)
    for element in axis:
        for i in np.arange(0,len(df),1):
            plt.plot(time,df[i]["attractor_position_%s" %element])
        plt.legend()
        plt.ylabel(r"%s_pos [AU]" %element)
        plt.xlabel("time [s]")
        #plt.yscale("log")
        #plt.savefig("%s_mean.png" %element, dpi=300, bbox_inches ="tight")
        if(save==True):
            save_as = save_folder + "attr_position/" + save_filename + "%s"%(element) + file_type
            plt.savefig(save_as, dpi=300,bbox_inches="tight")
        plt.show()
    return

def plot_hist_per_harmonic(df,var_of_interest="amplitude",axis="x",log=True,no_harmonics=15,save=False,save_folder=None,save_filename=None,file_type = ".png"):
    for k in np.arange(0,no_harmonics,1):
        for j in np.arange(0,len(df)):
            output_list=[]
            for i in range(len(df[0])):
                output_list.append(df[j]["%s_%s" %(var_of_interest,axis)][i][k])
            if(log==True):
                bins=np.logspace(np.log10(1e-6),np.log10(1e-2), 50)
                plt.xscale("log")
            else:
                bins=int(len(output_list)/100)
            plt.hist(output_list, bins=bins,alpha=0.5)
        if(save==True):
            save_as = save_folder + "histogram_%s/" %var_of_interest + save_filename + "%d"%(k+1) + file_type
        plt.title("Harmonic: %d"%(k+1))    
        plt.xlabel(r" %s NSD [a.u./$\sqrt{Hz}$]" %axis)    
        plt.ylabel("#")
        plt.savefig(save_as, dpi=300,bbox_inches="tight")
        plt.show()
    return




def plot_basics(folder_basic,Data_File_Number,axis='x',mode="psd_only",max_file=5,res=50000,fsamp=5000,save_files=True):
    folder_shaking = "/Shaking/" + "/Shaking%d/" % Data_File_Number
    folder = folder_basic + folder_shaking
    files = load_dir(folder, file_prefix = 'Shaking', max_file=max_file)
    if(mode=="psd_only"):
        for i in range(len(files)):
            data = files[i].xyz2
            data_det=signal.detrend(data)
            x=data_det[0]
            y=data_det[1]
            z=data_det[2]
            res = res
            fsamp = fsamp # stays hard coded for now
            xpsd, freqs = matplotlib.mlab.psd(x, Fs = fsamp, NFFT = res)
            ypsd, freqs = matplotlib.mlab.psd(y, Fs = fsamp, NFFT = res)
            zpsd, freqs = matplotlib.mlab.psd(z, Fs = fsamp, NFFT = res)
            if(axis=="x"):
                create_plot(freqs,xpsd)
            else: print("not implemented")    
        return freqs,xpsd 
    if(mode=="harmonics"):
        harmonic_list=[]
        for i in range(len(files)):
            data = files[i].xyz2
            data_det=signal.detrend(data)
            x=data_det[0]
            y=data_det[1]
            z=data_det[2]
            res = res
            fsamp = fsamp # stays hard coded for now
            xpsd, freqs = matplotlib.mlab.psd(x, Fs = fsamp, NFFT = res)
            ypsd, freqs = matplotlib.mlab.psd(y, Fs = fsamp, NFFT = res)
            zpsd, freqs = matplotlib.mlab.psd(z, Fs = fsamp, NFFT = res)
            freqs,harmonics = get_harmonics(axis,shake_freq=13,no_harmonics=10)
            harmonic_list.append(harmonics)
        return harmonic_list      