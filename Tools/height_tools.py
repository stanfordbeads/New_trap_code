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
#sys.path.append('/home/analysis_user/New_trap_code/Tools/')
import BeadDataFile
from discharge_tools import *
from analysis_tools import *
from joblib import Parallel, delayed
import multiprocessing



### load images in .bmp format
def load_img_files(path,print_list=True,max_files=10000):
    '''
    load ".bmp" files from a given path. Sorted by the digits.
    '''
    # read in the data 
    files = glob.glob(path)
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))    
    img_files = []
    if(len(files)<max_files):max_files=len(files) # set the maximum to the length
    for i in range(max_files):
        img_files.append(cv2.imread(files[i],0))
        if(print_list==True):print(files[i],i) 
    return img_files

def load_npy_files(path,print_list=True,max_files=10000):
    '''
    load ".npy" files from a given path. Sorted by the digits.
    '''
    files = glob.glob(path)
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))    
    img_files = []
    if(len(files)<max_files):max_files=len(files) # set the maximum to the length
    for i in range(max_files):
        img_files.append(np.load(files[i],0))
        if(print_list==True):print(files[i],i) 
    return img_files    

def laplacian(img):
    '''
    Calculate the laplacian of an image.
    '''
    lap = np.sum(cv2.Laplacian(img,cv2.CV_64F)*cv2.Laplacian(img,cv2.CV_64F))
    return lap

def threshold_max(img,threshold,lower_lim=600,upper_lim=800):
    '''
    Get the maximum bin of the thresholded image projected on the y_axis of the image. Sets values above threshold to 255 and below to 0. Uses cv2 method.
    '''
    ret,thresh = cv2.threshold(img.transpose()[lower_lim:upper_lim],threshold,255,0)
    return np.argmax(np.mean(thresh,axis=0))


def pixel_to_height(pixel,tot=1024,pix_res=4.6,mag=10,calibration=False,pix_size_from_calib=0.5):
    '''
    Get the height from a camera image in metric units. Pixel size and magnification are necessary as well as the resolution in one direction.
    '''
    total_pixel = tot # pixel
    pixel_resolution = pix_res # um from camera
    magnification = mag # have to check, whether this is the correct one
    if(calibration==False):pixel_per_um = total_pixel*pixel_resolution/magnification
    if(calibration==True):pixel_per_um = total_pixel * pix_size_from_calib    
    um = pixel_per_um - (pixel_per_um*pixel/total_pixel) # subtract as the 0 is upper left not bottom left corner, total_pixel cancels out, but for educational purpose
    return um

def center_of_mass(img,threshold):
    '''
    Threshold an image and calculate the center of mass position.
    '''
    ret,thresh = cv2.threshold(img,threshold,255,0)
    M = cv2.moments(thresh)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])  
    return cX,cY


def selected_areas_height_comparison(img,threshold,area_low_limits=[500,670,750],area_width=[100,40,100],pixel_or_height="height",plot=True,save_fig=False,output_path="",img_height=1024,color_list=["red","blue","green"]):
    ret,thresh = cv2.threshold(img,threshold,255,0)
    projection_list = []
    for i in range(len(area_low_limits)):
        projection_list.append(get_height_projection(thresh,low_lim=area_low_limits[i],up_lim=area_low_limits[i]+area_width[i],pixel_or_height=pixel_or_height,plot=plot))
    if(plot==True):
        if(pixel_or_height=="height"):plt.xlim(200,300)
        if(pixel_or_height=="pixel"):plt.xlim(400,600)  
        plt.show()
        if(save_fig==True):plt.savefig(output_path+"projection.png",dpi=300,bbox_inches="tight")
        plt.imshow(thresh)
        ax = plt.gca()
        # Add the patch to the Axes
        for i in range(len(area_low_limits)):
            p =patches.Rectangle((area_low_limits[i],0), area_width[i], img_height, angle=0.0,fill=False,color=color_list[i])
            ax.add_patch(p)    
        if(save_fig==True):plt.savefig(output_path+"selected_areas.png",dpi=300,bbox_inches="tight")
    return projection_list


############## No Notch Filter ################
def gaussian_bead_pos_fit(img,axis=0,low_x_lim=630,up_x_lim=700,low_y_lim=420,up_y_lim=550,upper_area=30000,up_lim_width=10,img_height=1024,img_width=1280):
    '''
    Gausian fit for the non-notch filtered images of the bead either reflected from shield/attractor or bead only
    '''
    
    # re-initialize chisquare, must be a more elegant way. However, no problem encountered so far doing it this way
    def chisquare_1d(function, functionparams, data_x, data_y,data_y_error):
        chisquarevalue=np.sum(np.power(np.divide(np.subtract(function(data_x,functionparams),data_y),data_y_error),2))
        ndf = len(data_y)-len(functionparams)
        #print(ndf)
        return (chisquarevalue, ndf)    
    
    def chisquare_gaussian(area,mean,sigma,constant):
        return chisquare_1d(function=gaussian,functionparams=[area,mean,sigma,constant],data_x=data_x,data_y=data_y,data_y_error=data_y_error)[0]
    
    
    # initialize values
    area=0
    mean=0
    sigma=0
    constant=0
    
    # pick axis you want to fit
    if(axis==0):
        img2 = img.transpose()
        data_x = range(img_height) # give x data
        data_y = np.mean(img2[low_x_lim:up_x_lim],axis=0) # give y data 
        data_y_error = np.sqrt(data_y)+1 # give y uncertainty
        low_lim_mean = low_y_lim # derive y limits for fit values
        up_lim_mean = up_y_lim # derive y limits for fit values        
        
    if(axis==1):
        data_x = range(img_width) # give x data
        data_y = np.mean(img[low_y_lim:up_y_lim],axis=0) # give y data 
        data_y_error = np.sqrt(data_y)+1 # give y uncertainty
        low_lim_mean = low_x_lim # derive x limits for fit values  
        up_lim_mean = up_x_lim # derive x limits for fit values  
              
    m=Minuit(chisquare_gaussian, 
             area = 100, # set start parameter
             error_area = 1,
             limit_area= (0,upper_area), # limit to a value so it does not fit weird things
             #fix_area = "True", # you can also fix it
             mean = np.argmax(data_y), # a good start guess in this method is the larges pixel
             error_mean = 1,
             #fix_mean = "False",
             limit_mean = (low_lim_mean,up_lim_mean), # the mean should be somewhere, where you expect the bead
             sigma = 5, # the width should be usually within the limits 
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

def get_height_projection(file,low_lim=600,up_lim=700,pixel_or_height="height",plot=True,img_height=1024,pixel_size=0.503):
    """
    Calculate height projection for a given image. Optional you can show the reuslts in pixels or height
    """
    image = file.transpose()
    z = np.mean(image[low_lim:up_lim],axis=0)
    
    if(plot==True):
        if(pixel_or_height=="height"):x= img_height*pixel_size-np.arange(0,img_height*pixel_size,pixel_size)
        elif(pixel_or_height=="pixel"):x=range(img_height)
        plt.plot(x,z)
        plt.ylabel("mean intensity")
        if(pixel_or_height=="height"):plt.xlabel("height [um]")
        elif(pixel_or_height=="pixel"):plt.xlabel("pixel")
        plt.legend()
        #plt.show()  
        
    return z

def get_height_fit(file,low_x_lim=600,up_x_lim=700,low_y_lim=420,up_y_lim=550,up_lim_width=8,upper_area=30000,plot=True,img_height=1024,img_width=1280):
    '''
    Return fit function for Gaussian fit of the bead image without notch filter.
    '''
    img1 = file
    img2 = file.transpose()
    z = np.mean(img2[low_x_lim:up_x_lim],axis=0)   
    m = gaussian_bead_pos_fit(img1,axis=0,low_x_lim=low_x_lim,up_x_lim=up_x_lim,low_y_lim=low_y_lim,up_y_lim=up_y_lim,up_lim_width=up_lim_width,upper_area=upper_area,img_height=img_height,img_width=img_width)             
    if(plot==True):
        plt.plot(range(img_height),gaussian(range(img_height),params=[m[1].values["area"],m[1].values["mean"],m[1].values["sigma"],m[1].values["constant"]]),label="fit")
        plt.plot(range(img_height),z,label = "data")
        plt.xlabel("pixel")
        plt.ylabel("mean intensity")
        plt.legend()
        #plt.show()    
    return m

def gaussian_fit_shadow_height(img,low_x_lim=670,up_x_lim=710,low_y_lim=400,up_y_lim=750,upper_area=3000,up_lim_width=10,img_type="Image"):    
    def chisquare_1d(function, functionparams, data_x, data_y,data_y_error):
        chisquarevalue=np.sum(np.power(np.divide(np.subtract(function(data_x,functionparams),data_y),data_y_error),2))
        ndf = len(data_y)-len(functionparams)
        #print(ndf)
        return (chisquarevalue, ndf)    
    def chisquare_gaussian(area,mean,sigma,constant):
        return chisquare_1d(function=gaussian,functionparams=[area,mean,sigma,constant],data_x=data_x,data_y=data_y,data_y_error=data_y_error)[0]
        
    fit_range_low = low_y_lim
    fit_range_high = up_y_lim
        
    if(img_type=="Image"):
        img2 = img.transpose()
        data_x = np.arange(fit_range_low,fit_range_high,1)
        data_y = np.mean(img2[low_x_lim:up_x_lim],axis=0)[fit_range_low:fit_range_high] # give y data 
        constant=255
    if(img_type=="Projection"): 
        data_x = np.arange(fit_range_low,fit_range_high,1)
        data_y = img[fit_range_low:fit_range_high]
        constant=255
    if(img_type=="Diff_Projection"): 
        data_x = range(len(img)) # give x data
        data_y = img
        constant = 0
        

    data_y_error = np.sqrt(np.abs(data_y))+1 # give y uncertainty
    low_lim_mean = low_y_lim
    up_lim_mean = up_y_lim

    m=Minuit(chisquare_gaussian, 
             area = -1000, # set start parameter
             error_area = 1,
             limit_area= (-upper_area,-100), # if you want to limit things
             #fix_area = "False", # you can also fix it
             mean = (low_y_lim+up_y_lim)/2,
             error_mean = 1,
             #fix_mean = "True",
             limit_mean = (low_lim_mean,up_lim_mean),
             sigma = 4,
             error_sigma = 1,
             limit_sigma=(0,up_lim_width),
             constant = constant,
             error_constant = 1,
             fix_constant="True",
             errordef = 1,
             print_level=0)
    #print('Now proceed with the fit.')
    m.migrad(ncall=500000)
    #m.minos(), if you need fancy mapping
    chisquare=m.fval
    #print(np.median(data_y))
    return m


def threshold_image(img,lower,upper):
    img2 = img.copy()
    mask = (lower < img2) & (img2 < upper)
    img2[mask] = 255
    img2[~mask] = 0
    return img2

def from_shadow_image_to_height(image,threshold,area_low_limits=[670,730],area_widths=[40,70],flb=460,fub=510,area_max=3000,width_max=5,plot=False):
    thresh = threshold_image(image.copy(),threshold,256)
    img = thresh.transpose()
    z1 = np.mean(img[area_low_limits[0]:area_low_limits[0]+area_widths[0]],axis=0)
    z2 = np.mean(img[area_low_limits[1]:area_low_limits[1]+area_widths[1]],axis=0)
    fit_img = z1-z2
    m = gaussian_fit_shadow_height(fit_img,low_y_lim=flb,up_y_lim=fub,upper_area=area_max,up_lim_width=width_max,img_type="Diff_Projection",)
    
    if(plot==True):
        plt.plot(fit_img)
        plt.plot(range(1024),gaussian(range(1024),params=[m.values["area"],m.values["mean"],m.values["sigma"],m.values["constant"]]),label="fit")
        plt.xlim(m.values["mean"]-100,m.values["mean"]+100)
    # height =  pixel_to_height(m.values["mean"],calibration=calibration)        
    return m.values["mean"],m # in pixels


# add tools for the Y position

def gaussian_fit_shadow_width(img,low_x_lim=670,up_x_lim=710,low_y_lim=350,up_y_lim=650,upper_area=3000,up_lim_width=10,img_type="Image"):    
    def chisquare_1d(function, functionparams, data_x, data_y,data_y_error):
        chisquarevalue=np.sum(np.power(np.divide(np.subtract(function(data_x,functionparams),data_y),data_y_error),2))
        ndf = len(data_y)-len(functionparams)
        #print(ndf)
        return (chisquarevalue, ndf)    
    def chisquare_gaussian(area,mean,sigma,constant):
        return chisquare_1d(function=gaussian,functionparams=[area,mean,sigma,constant],data_x=data_x,data_y=data_y,data_y_error=data_y_error)[0]
        

    fit_range_low = low_x_lim
    fit_range_high = up_x_lim
        
    if(img_type=="Image"):
        data_x = np.arange(0,1280,1)
        data_y = np.mean(img[low_y_lim:up_y_lim],axis=0)# give y data 
        constant=255
    if(img_type=="Projection"): 
        data_x = np.arange(fit_range_low,fit_range_high,1)
        data_y = img[fit_range_low:fit_range_high]
        constant=255
    if(img_type=="Diff_Projection"): 
        data_x = range(len(img)) # give x data
        data_y = img
        constant = 0
        
    data_y_error = np.sqrt(np.abs(data_y))+1 # give y uncertainty
    low_lim_mean = low_y_lim
    up_lim_mean = up_y_lim

    m=Minuit(chisquare_gaussian, 
             area = -1000, # set start parameter
             error_area = 1,
             limit_area= (-upper_area,-100), # if you want to limit things
             #fix_area = "False", # you can also fix it
             mean = (low_lim_mean+up_lim_mean)/2,
             error_mean = 1,
             #fix_mean = "True",
             limit_mean = (low_lim_mean,up_lim_mean),
             sigma = 4,
             error_sigma = 1,
             limit_sigma=(0,up_lim_width),
             constant = 0,
             error_constant = 1,
             fix_constant="False",
             errordef = 1,
             print_level=0)
    #print('Now proceed with the fit.')
    m.migrad(ncall=500000)
    #m.minos(), if you need fancy mapping
    chisquare=m.fval
    #print(np.median(data_y))
    return m

def from_shadow_image_to_width(image,threshold,area_low_limits=[650,750],area_width=70,flb=0,fub=70,area_max=3000,width_max=5,plot=False):
    thresh = threshold_image(image.copy(),threshold,256)
    img = thresh
    low_bound_y=600
    up_bound_y=700
    z1 = np.mean(img[low_bound_y:up_bound_y],axis=0)[area_low_limits[0]:area_low_limits[0]+area_width]
    #plt.plot(z1)
    z2 = np.mean(img[low_bound_y:up_bound_y],axis=0)[area_low_limits[1]:area_low_limits[1]+area_width]
    fit_img = z1-z2
    m = gaussian_fit_shadow_width(fit_img,low_y_lim=flb,up_y_lim=fub,upper_area=area_max,up_lim_width=width_max,img_type="Diff_Projection",)
    #print(m.values["area"],m.values["mean"],m.values["sigma"],m.values["constant"])
    if(plot==True):
        plt.plot(fit_img)
        plt.plot(range(area_width),gaussian(range(area_width),params=[m.values["area"],m.values["mean"],m.values["sigma"],m.values["constant"]]),label="fit")
#plt.xlim(m.values["mean"]-100,m.values["mean"]+100)
    return m.values["mean"],m


## everything for beam profiling

def gaussian_beam(x,params=list):
    '''
    Generic defintion of a Gaussian bead profile (from Akio)
    '''
    #norm = (1/((1/2*params[2])*np.sqrt(np.pi * 2)))
    return params[0] * np.exp(-2*(np.subtract(x,params[1])**2/(params[2]**2)))+params[3]


def beam_width(x,params=list,wave_length=1.064):
    #defines the beam width using 1064 nm laser
    return params[0]*np.sqrt(1+((x-params[1])/(np.pi*params[0]*params[0]/wave_length))**2)


def prepare_profile_data(files):
    # prepare the data into a data frame using the calibration Akio has performed
    df = pd.DataFrame()
    spin_list = [] # a list for the spin_sum
    qsum_list = [] # a list for the quad_sum 
    cant_pos_list_x, cant_pos_list_y, cant_pos_list_z = [[] for  x in range(3)] # save the positions of the cantielever
    spin_down_size_factor = len(files[0].spin_data)/len(files[0].xyz[0]) # 10 for normal operation, but can be different
    
    for i in tqdm(range(len(files))):
        spin = np.zeros(len(files[0].xyz[0]))
        spin_temp = files[i].spin_data
        for j in range(len(files[i].xyz[0])):
            spin[j]=np.average(spin_temp[int(spin_down_size_factor)*j: int(spin_down_size_factor)*j+ int(spin_down_size_factor-1)]) # average down to 5000 in order to match spin with the cantilever position.    
        spin_list.append(spin)
        qsum_list.append(files[i].quad_sum)
        cant_pos_list_x.append(files[i].cant_pos[0])
        cant_pos_list_y.append(files[i].cant_pos[1])
        cant_pos_list_z.append(files[i].cant_pos[2])
    df["QPD_SUM"] = qsum_list
    df["SPIN_SUM"] = spin_list 
    df["CANT_POS_X"] = cant_pos_list_x
    df["CANT_POS_Y"] = cant_pos_list_y
    df["CANT_POS_Z"] = cant_pos_list_z
    
    
    # use calibration
    ## applies calibration assuming x and y have the same
    df["CANT_POS_X_cal"] = df["CANT_POS_X"].apply(lambda element: voltage_to_x_position(element))  
    df["CANT_POS_Y_cal"] = df["CANT_POS_Y"].apply(lambda element: voltage_to_x_position(element)) 
    df["CANT_POS_Z_cal"] = df["CANT_POS_Z"].apply(lambda element: voltage_to_z_position(element))
    
    
    # differentiate the power to get dP/dx plots
    df["QPD_SUM_diff"] = df["QPD_SUM"].apply(lambda element: np.diff(element)) # 
    df["SPIN_SUM_diff"] = df["SPIN_SUM"].apply(lambda element: np.diff(element)) # 

    return df