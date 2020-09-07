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

def chisquare_gaussian_test(area,mean,sigma,constant,data_x,data_y,data_y_error):
    return chisquare_1d(function=gaussian,functionparams=[area,mean,sigma,constant],data_x=data_x,data_y=data_y,data_y_error=data_y_error)[0]


def chisquare_linear(a,b):
    return chisquare_1d(function=linear,functionparams=[a,b],data_x=data_x,data_y=data_y,data_y_error=data_y_error)[0]


# calibration of the au into power as shown by Akio on 20200707
def au_to_power (x,p0 = -0.02715,p1 = -1.437e-5):
    power_mW = x*p1 + p0
    return power_mW

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

## estimate the stroke
def estimate_stroke(cant_pos_y):
    stroke = voltage_to_x_position(np.std(cant_pos_y)*np.sqrt(2))
    return 2*stroke


# 



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
    freqs=np.linspace(0,int(fsamp/2),(int(res/2))+1)    # change
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
    freqs=np.linspace(0,int(fsamp/2),(int(res/2))+1)    # change
    spin_FFT=np.fft.rfft(data_det)
    norm = np.sqrt(2 / (res* fsamp))
    spin_nsd = norm**2 * (spin_FFT * spin_FFT.conj()).real
    spin_phase=np.angle(spin_FFT)
    return freqs,spin_nsd,spin_phase


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