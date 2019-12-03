import discharge_tools
import numpy as np
import h5py
import matplotlib.pyplot as plt
import time
import pandas as pd
plt.rcParams["figure.figsize"] = (12,9)
plt.rcParams["font.size"] = 24
plt.rcParams['xtick.labelsize'] = "small"
plt.rcParams['ytick.labelsize'] = 36

def get_temperature_and_pressure(dates):
#         if(i<12):
#             am_or_pm="AM"
#         if(i>=12):    
#             am_or_pm="PM"  # OLDFORMAT before 20191121
#         if(i==0):
#             i = i+12
#         if(i>12):
#             i = (i-12)


# x_value = (np.arange(13,24,1/(60*60)))
#plt.plot(s)
#plt.ylabel("Temperature [C]",fontsize=24)
#plt.xlabel("seconds",fontsize=24)
#plt.xticks(fontsize=16)
#plt.yticks(fontsize=16)
#plt.savefig("s_temperature_cycle_new_201391125_0_9.png", dpi = 250, bbox_inches ="tight")

    airtemperature_list, surfacetemperature_list, pressure_list, f=([] for i in range(4))
    for date in dates:
        print(date)
        for i in np.arange(0,24):
            try:
                if(i<10):
                    hour = "0%d" %i
                if(i>9):
                    hour ="%d" %i
                f.append(h5py.File("/data/SC_data/TemperatureAndPressure%s/TempAndPressure%s_%s.hdf5" %(date,date,hour), mode='r+'))
            except:
                print("%s hour at %s is not on record" %(i,date))
                continue       
    for i in np.arange(0,len(f),1):
        try: 
            None
            airtemperature_list.extend(list(f[i]["AirTemperature/AirTemperatures"]))
            surfacetemperature_list.extend(list(f[i]["SurfaceTemperature/SurfaceTemperatures"]))
            pressure_list.extend(list(f[i]["Pressure/Pressures"]))
        except:
            continue
    #print(lst_dict_temp)        
    #df.append(lst_dict_temp)       
    [f_.close() for f_ in f] ## good programing       
    return airtemperature_list,surfacetemperature_list, pressure_list

def get_harmonics(input_psd,shake_freq,no_harmonics,res_factor=10,plot=False):
    harmonics = []
    FreqTF = np.arange(shake_freq,(shake_freq*no_harmonics+shake_freq),shake_freq)
    for i in range(no_harmonics):
        harmonics.append(input_psd[shake_freq*res_factor*(i+1)])
    if(plot==True):
        plt.plot(FreqTF,harmonics, marker ="o", linestyle ="")
        plt.yscale("log")
        #plt.show()
    return FreqTF,harmonics     

def get_mean_std_sum_of_harmonics(df,no_harmonics=10):
    mean_list, std_list, sum_list, median_list=([] for i in range(4))

    for i in np.arange(0,no_harmonics,1):
        std_list.append(np.std(df["%d" %i]))
        mean_list.append(np.mean(df["%d" % i])) 
        sum_list.append(np.sum(df["%d" %i]))
        median_list.append(np.median(df["%d" %i]))
    return mean_list,std_list,sum_list,median_list   

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