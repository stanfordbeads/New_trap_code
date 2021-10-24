import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
import matplotlib.patches as patches


from matplotlib import mlab as mlab
from matplotlib.mlab import psd

import gif

import os

import cv2

import scipy
from scipy import signal

import h5py

import time
import datetime as dt
from tqdm import tqdm_notebook as tqdm

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
from processor_tools import *

from joblib import Parallel, delayed
import multiprocessing

from bisect import bisect_left
from itertools import dropwhile, islice


from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from scipy.ndimage import fourier_shift
from skimage.filters import window
