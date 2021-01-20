import numpy as np
import matplotlib.pyplot as plt

from likelihood_calculator import likelihood_analyser
import BeadDataFile

import LeastSquares

print('************************')
print('Starting simple sine fit')
print('************************')

lc_i = likelihood_analyser.LikelihoodAnalyser()

# generating fake data
samples = 5000
noise_rms = 1
time_x = np.arange(0, samples) / 5000
freq = 100
amp = 1
phi = 1
sig_x = amp * np.sin(2 * np.pi * freq * time_x) + noise_rms * np.random.randn(samples)
sig_x2 = amp * np.sin(2 * np.pi * freq * time_x + phi) + noise_rms * np.random.randn(samples)

# x = np.random.randn(samples)

# fit arguments
fit_kwargs = {'A': 5, 'f': 100, 'phi': 0, 'sigma': 1,
              'error_A': 2, 'error_f': 10, 'error_phi': 0.5, 'error_sigma': 0.1, 'errordef': 1,
              'limit_phi': [0, 2 * np.pi],
              'print_level': 0, 'fix_f': True, 'fix_phi': False, 'fix_sigma': True}

fit_kwargs = {'A': 5, 'f': 100, 'phi': 0,
              'error_A': 2, 'error_f': 10, 'error_phi': 0.5, 'errordef': 1,
              'limit_phi': [0, 2 * np.pi],
              'print_level': 0, 'fix_f': True, 'fix_phi': False}

empirical_scale = 0.18642853643227691
m1 = lc_i.find_mle_sin(sig_x, drive_freq=freq, noise_rms=noise_rms*empirical_scale, bandwidth=100, plot=True,
                       suppress_print=False, **fit_kwargs)

fit_kwargs = {'A': 5, 'f': 100, 'phi': 1, 'A2': 1, 'f2': 100, 'delta_phi': 1,
              'error_A': 2, 'error_f': 10, 'error_phi': 0.5, 'errordef': 1,
              'error_A2': 2, 'error_f2': 10, 'error_delta_phi': 0.5,
              'limit_phi': [0, 2 * np.pi],
              'print_level': 0, 'fix_f': True, 'fix_phi': False, 'fix_f2': True, 'fix_delta_phi': True, 'fix_A2': False}

empirical_scale = 0.18642853643227691
m1 = lc_i.find_mle_2sin(sig_x, sig_x2, drive_freq=freq, noise_rms=noise_rms * empirical_scale, bandwidth=100, plot=True,
                        suppress_print=False, **fit_kwargs)