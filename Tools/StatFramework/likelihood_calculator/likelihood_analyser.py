import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from iminuit import Minuit
import LeastSquares
import time


class LikelihoodAnalyser:
    def __init__(self):
        self.data_x = 0  # x-data to fit
        self.data_y = 0  # y-data to fit
        self.fsamp = 0  # sampling rate
        self.noise_sigma = 1  # gaussian white noise std

    def least_squares_sine(self, A, f, phi):
        """
        least squares for minimization - sine function
        :param A: Amplitude
        :param f: frequency
        :param phi: phase
        :return: cost function - sum of squares
        """
        func_t = A * np.sin(2 * np.pi * f * self.data_x + phi)  # function to minimize
        res = sum(np.power(np.abs(self.data_y - func_t), 2))
        return res / self.noise_sigma ** 2

    def least_squares_sine2(self, A, f, phi, sigma):
        """
        least squares for minimization - sine function
        This function takes the white noise width as a parameter as well
        :param sigma: gaussian white noise variance
        :param A: Amplitude
        :param f: frequency
        :param phi: phase
        :return: cost function - sum of squares
        """
        func_t = A * np.sin(2 * np.pi * f * self.data_x + phi)  # function to minimize
        res = sum(np.power(np.abs(self.data_y - func_t), 2))
        res += len(self.data_x) * np.log(np.pi * sigma ** 2)
        return res / sigma ** 2

    def find_mle_sin(self, x, drive_freq=0, fsamp=5000, bandwidth=50, noise_rms=0, plot=False, suppress_print=True,
                     **kwargs):
        """
        The function is fitting the data with a sine template using iminuit.
        The fitting is done after applying a bandpass filter.
        :param suppress_print: suppress all printing
        :param noise_rms: std of the white gaussian noise
        :param plot: plot the data and its fft
        :param bandwidth: bandwidth for butter filter [Hz]
        :param fsamp: sampling rate [1/sec]
        :param x: 1 dim. position data (time domain)
        :param drive_freq: drive frequency of the response
        :return: estimated values, chi_square
        """
        if not suppress_print:
            print('Data overall time: ', len(x) / fsamp, ' sec.')
        self.fsamp = fsamp
        if noise_rms != 0:
            self.noise_sigma = noise_rms

        # apply a bandpass filter to data and store data in the correct place for the minimization
        self.data_x = np.arange(0, len(x)) / fsamp
        start = time.time()
        if drive_freq != 0:
            if not suppress_print:
                print('Bandpass filter ON. Bandwidth: ', bandwidth, 'Hz')
            b, a = signal.butter(3, [2. * (drive_freq - bandwidth / 2.) / self.fsamp,
                                     2. * (drive_freq + bandwidth / 2.) / self.fsamp], btype='bandpass')
            self.data_y = signal.filtfilt(b, a, x)
        else:
            self.data_y = x
        end = time.time()
        if not suppress_print:
            print('bandpass time: ', end - start)

        # we create an instance of Minuit and pass the function to minimize
        if 'sigma' in kwargs.keys():
            mimuit_minimizer = Minuit(self.least_squares_sine2, **kwargs)
        else:
            mimuit_minimizer = Minuit(self.least_squares_sine, **kwargs)

        start = time.time()
        mimuit_minimizer.migrad(ncall=50000)
        end = time.time()
        if not suppress_print:
            print('minimization time: ', end - start)
            print(mimuit_minimizer.get_param_states())

        if plot:
            _, ax = plt.subplots(1, 2, figsize=(9.5, 4))
            ax[0].scatter(self.data_x, self.data_y)
            fft = np.abs(np.fft.rfft(x)) ** 2
            freq = np.fft.rfftfreq(len(x), d=1. / fsamp)
            ax[0].set(title='raw data')
            plt.subplot(122)
            mimuit_minimizer.draw_profile('A', subtract_min=True)
            plt.show()
        if not suppress_print:
            print('reduced chi2: ', mimuit_minimizer.fval / (len(self.data_y) - 2))

        return mimuit_minimizer
