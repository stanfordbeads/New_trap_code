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



### load images in .bmp format
def load_img_files(path,print_list=True):
    # read in the data 
    files = glob.glob(path)
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))    
    img_files = []
    for i in range(0,len(files)):
        img_files.append(cv2.imread(files[i],0))
        if(print_list==True):print(files[i]) 
    return img_files

def load_npy_files(path,print_list=True):
    files = glob.glob(path)
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))    
    img_files = []
    for i in range(0,len(files)):
        img_files.append(np.load(files[i],0))
        if(print_list==True):print(files[i]) 
    return img_files    

def load_dir_reduced_to_height(dirname,file_prefix,max_files):
    '''
    
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

def laplacian(img):
    lap = np.sum(cv2.Laplacian(img,cv2.CV_64F)*cv2.Laplacian(img,cv2.CV_64F))
    return lap

def threshold_max(image,threshold):
    image_to_t = image.transpose()[600:800]
    ret,thresh = cv2.threshold(image_to_t,threshold,255,0)
    return np.argmax(np.mean(thresh,axis=0))


def pixel_to_height(pixel,tot=1024,pix_res=4.6,mag=10):
    total_pixel = tot # pixel
    pixel_resolution = pix_res # um
    magnification = mag # have to check, whether this is the correct one
    pixel_per_um = total_pixel*pixel_resolution/magnification
    um = pixel_per_um - (pixel_per_um*pixel/total_pixel) # subtract as the 0 is upper left not bottom left corner, total_pixel cancels out, but for educational purpose
    return um

def center_of_mass(image,threshold):
    ret,thresh = cv2.threshold(image,threshold,255,0)
    M = cv2.moments(thresh)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])  
    return cX,cY

def gaussian(x,params=list):
    norm = (1/((1/2*params[2])*np.sqrt(np.pi * 2)))
    return params[0] * norm * np.exp(-(np.subtract(x,params[1])**2/(2*params[2]**2)))+params[3]

def gaussian_bead_pos_fit(img,axis=0,low_x_lim=630,up_x_lim=700,low_y_lim=420,up_y_lim=550,upper_area=30000,up_lim_width=10):
    def chisquare_1d(function, functionparams, data_x, data_y,data_y_error):
        chisquarevalue=np.sum(np.power(np.divide(np.subtract(function(data_x,functionparams),data_y),data_y_error),2))
        ndf = len(data_y)-len(functionparams)
        #print(ndf)
        return (chisquarevalue, ndf)    
    def chisquare_gaussian(area,mean,sigma,constant):
        return chisquare_1d(function=gaussian,functionparams=[area,mean,sigma,constant],data_x=data_x,data_y=data_y,data_y_error=data_y_error)[0]
    
    area=0
    mean=0
    sigma=0
    constant=0
    
    if(axis==0):
        img2 = img.transpose()
        data_x = range(1024) # give x data
        data_y = np.mean(img2[low_x_lim:up_x_lim],axis=0) # give y data 
        data_y_error = np.sqrt(data_y)+1 # give y uncertainty
        low_lim_mean = low_y_lim
        up_lim_mean = up_y_lim
        
    if(axis==1):
        data_x = range(1280) # give x data
        data_y = np.mean(img[low_y_lim:up_y_lim],axis=0) # give y data 
        data_y_error = np.sqrt(data_y)+1 # give y uncertainty
        low_lim_mean = low_x_lim
        up_lim_mean = up_x_lim
              
    m=Minuit(chisquare_gaussian, 
             area = 100, # set start parameter
             error_area = 1,
             limit_area= (0,upper_area), # if you want to limit things
             #fix_area = "True", # you can also fix it
             mean = np.argmax(data_y),
             error_mean = 1,
             #fix_mean = "False",
             limit_mean = (low_lim_mean,up_lim_mean),
             sigma = 15,
             error_sigma = 1,
             limit_sigma=(0,up_lim_width),
             constant = 0,
             error_constant = 1,
             #fix_constant=0,
             errordef = 1,
             print_level=0)
    #print('Now proceed with the fit.')
    m.migrad(ncall=500000)
    #m.minos(), if you need fancy mapping
    chisquare=m.fval
    return m.values['mean'],m

def show_height_projection(file,low_lim=600,up_lim=700,pixel_or_height="height"):
    image = file.transpose()
    z = np.mean(image[low_lim:up_lim],axis=0)
    if(pixel_or_height=="height"):x= 1024*0.46-np.arange(0,1024*0.46,0.46)
    elif(pixel_or_height=="pixel"):x=range(1024)
    plt.plot(x,z)
    plt.ylabel("mean intensity")
    if(pixel_or_height=="height"):plt.xlabel("height [um]")
    elif(pixel_or_height=="pixel"):plt.xlabel("pixel")
    plt.legend()
    plt.show()
    return z

def show_height_fit(file,low_x_lim=600,up_x_lim=700,low_y_lim=420,up_y_lim=550,up_lim_width=8,upper_area=30000):
    img1 = file
    img2 = file.transpose()
    z = np.mean(img2[low_x_lim:up_x_lim],axis=0)       
    m=gaussian_bead_pos_fit(img1,axis=0,low_x_lim=low_x_lim,up_x_lim=up_x_lim,low_y_lim=low_y_lim,up_y_lim=up_y_lim,up_lim_width=up_lim_width,upper_area=upper_area)
    plt.plot(range(1024),gaussian(range(1024),params=[m[1].values["area"],m[1].values["mean"],m[1].values["sigma"],m[1].values["constant"]]),label="fit")
    plt.plot(range(1024),z,label = "data")
    plt.xlabel("pixel")
    plt.ylabel("mean intensity")
    plt.legend()
    #plt.show()
    return m