
# coding: utf-8

# ### Notebook to create the utility file for the signal model input ###

#### Import
import numpy as np
import pickle as pkl
import scipy.interpolate as interp
import scipy, sys, time
from bisect import bisect_left
sys.path.append('/home/analysis_user/New_trap_code/Tools/')
import BeadDataFile
from discharge_tools import load_dir
lambdas = np.logspace(-6.3, -3, 100)
sep_list = np.arange(0.0e-6,100e-6,0.5e-6)
height_list = np.arange(-15.0e-6,15.0e-6,0.5e-6)
### define functions


def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before



### part 1:load files

## load the data dictionary file (its usually of the form results_dic[rbead][sep][height][yuklambda])

def load_file(separation,height,lambda_par=1e-5,alpha=1):
    """
Load position and force from a file based on separation, height and the lambda_parameter. Alpha only scales the result. If not existing it picks the closest parameter in the list. The original list it got greated from are:
lambdas = np.logspace(-6.3, -3, 100) for lambdas
sep_list = np.arange(0.0e-6,100e-6,0.5e-6) for separation
height_list = np.arange(-15.0e-6,15.0e-6,0.5e-6) for height
    """ 
    try:
        #print(height)
        res_dict_side_by_side = pkl.load( open('/home/analysis_user/New_trap_code/SensitivityFramework/results/simulation/rbead_2.4e-06_sep_%4.1e_height_%4.1e.p' % (separation,height) ,'rb'))
    except:
        print("Your choice of separation or height is not existing")
        val2 = take_closest(sep_list, separation)
        val3 = take_closest(height_list, height)
        separation=val2  
        height=val3
        print("Taking %4.1e for separation" %val2)
        print("Taking %4.1e for height" %val3)
        res_dict_side_by_side = pkl.load( open('/home/analysis_user/New_trap_code/SensitivityFramework/results/simulation/rbead_2.4e-06_sep_%4.1e_height_%4.1e.p' %(separation,height), 'rb'))
    try:
        res_dict_side_by_side[2.4e-6][separation][height][lambda_par][0]        
    except:
        print("Your choice of lambda is not existing")
        val = take_closest(lambdas, lambda_par)
        lambda_par=val  
        print("Taking %2.2e for lambda" %val)
    for item in res_dict_side_by_side[2.4e-6]:
        print("A separation of %2.2e is selected" %item)
        separation=item # as separation is saved differently here than in the file name
    for item2 in res_dict_side_by_side[2.4e-6][separation]:
        #print(res_dict_side_by_side[2.4e-6][separation])
        height=item2
        print("A height of %2.2e is selected" %item2)      
    force_x = res_dict_side_by_side[2.4e-6][separation][height][lambda_par][0] # force in direction of the sphere
    force_y = res_dict_side_by_side[2.4e-6][separation][height][lambda_par][1] # force in direction perpendicular to the sphere
    force_z = res_dict_side_by_side[2.4e-6][separation][height][lambda_par][2] # force in z-direction
    force_x_yuk = alpha*res_dict_side_by_side[2.4e-6][separation][height][lambda_par][3] # force by the yukawa potential , x
    force_y_yuk = alpha*res_dict_side_by_side[2.4e-6][separation][height][lambda_par][4] # force by the yukawa potential , y
    force_z_yuk = alpha*res_dict_side_by_side[2.4e-6][separation][height][lambda_par][5] # force by the yukawa potential , z
    pos = res_dict_side_by_side["posvec"] # get the position of the bead from the dictionary
    force_list = [force_x,force_y,force_z,force_x_yuk,force_y_yuk,force_z_yuk]
    return pos,force_list

### part 2: conversion of movement between the domains
# determine the center position of the attractor at a given time

def force_at_position(direction,pos,force_list,yuk_or_grav="yuk"):
    """ 
    define if pure gravity or pure yukawa is the force of interest and select the direction you want to use
    """
    if(yuk_or_grav=="yuk"):
        if(direction=="x"):
            force_vec = force_list[3]
        if(direction=="y"):
            force_vec = force_list[4]
        if(direction=="z"):
            force_vec = force_list[5]
    if(yuk_or_grav=="grav"):
        if(direction=="x"):
            force_vec = force_list[0]
        if(direction=="y"):
            force_vec = force_list[1]
        if(direction=="z"):
            force_vec = force_list[2]        
    return force_vec

# define a time with a sampling frequency over which the force is sampled. For simplicity 5000, our usual sampling frequency is taken and everything is normalized to a second
int_time =1 
sampling_frequency = 5000 # should be 5000 
time = np.arange(0,int_time,1/sampling_frequency) # make a time array

# sine
def position_at_time_sin_function(stroke,time,frequency,offset_y=0): 
    '''
    get the position for a given frequency and stroke using a pure sine wave. Included an offset in the y direction. 0 is the center of the attractor, which is center of the central gold finger
    '''
    pos_at_time = stroke/2*np.sin(2*np.pi*time*frequency)+offset_y
    return pos_at_time

# triang
def position_at_time_tri_function(stroke,time,frequency,width=0.5):
    '''
    get the position for a given frequency and stroke using a sawtooth wave. Included an offset in the y direction. 0 is the center of the attractor, which is center of the central gold finger    
    '''
    pos_at_time=stroke/2*signal.sawtooth(2 * np.pi * frequency * time+np.pi/2,width=width)
    return pos_at_time

# determine the force for a given point in time using the transformation to position

## sinusoidal movement
def force_at_a_time_sin_function(stroke,time,frequency,pos_vec,force_vec,offset_y=0):
    '''
    Interpolates between the position to get a smooth force vs time for a sine function
    '''
    osci_pos = position_at_time_sin_function(stroke,time,frequency,offset_y=offset_y)
    return np.interp(osci_pos,pos_vec,force_vec, left=None, right=None, period=None)

## triangle movement
def force_at_a_time_tri_function(stroke,time,frequency,pos_vec,force_vec,width=0.5):
    '''
    Interpolates between the position to get a smooth force vs time for a sawtooth function
    '''
    osci_pos_triang =  position_at_time_tri_function(stroke,time,frequency,width=width)
    return np.interp(osci_pos_triang,pos_vec,force_vec, left=None, right=None, period=None)

# use those two for most of your applications

def force_vs_position(separation,height,direction,lambda_par,yuk_or_grav="yuk",alpha=1):
    '''
    In order to be able also to implement own strokes and attractor movement profiles this extracts the pure force vs position of the attractor
    '''
    pos,force_list = load_file(separation,height,lambda_par,alpha)
    force = force_at_position(direction,pos,force_list,yuk_or_grav)
    return pos,force

def force_vs_time(separation,height,stroke,frequency,direction,lambda_par,offset_y=0,yuk_or_grav="yuk",alpha=1):
    '''
    This gives the force as a function of time for a sinusoidial movement in the y-direction. The time parameter is a second sampled with 5kHz.
    '''
    pos,force_list = load_file(separation,height,lambda_par,alpha)
    force_vec = force_at_position(direction,pos,force_list,yuk_or_grav="yuk")
    force = force_at_a_time_sin_function(stroke,time,frequency,pos,force_vec,offset_y=offset_y)
    return time,force