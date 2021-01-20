import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from iminuit import Minuit
import time


class LikelihoodAnalyser:
    def __init__(self):
        self.data_x = 0  # x-data to fit
        self.data_y = None  # y-data to fit
        self.data_y2 = 0  # second y-data to fit
        self.fsamp = 5000  # sampling rate
        self.noise_sigma = 1  # gaussian white noise std
        self.noise_sigma2 = 1  # gaussian white noise std

        self.template = None
        self.template2 = None

        self.harmoincs_amp = None
        self.harmoincs_phases = None
        self.harmoincs_freqs = None
        self.harmoincs_noise = None

        self.harmoincs_amp2 = None
        self.harmoincs_phases2 = None
        self.harmoincs_amp3 = None
        self.harmoincs_phases3 = None
        self.harmoincs_amp4 = None
        self.harmoincs_phases4 = None

    def log_likelihood_template(self, alpha, phase, sigma):
        """
        Log likelihood function, using template and control dataset to constrain the noise
        :param alpha: scale factor
        :param phase: phase of the template
        :param sigma: noise
        :return: -2log(likelihood)
        """
        func_t = alpha * np.array(self.template)  # function to minimize
        func_t = np.roll(func_t, int(phase))

        res = sum(np.power(np.abs(self.data_y - func_t), 2)) / sigma ** 2
        res += sum(np.power(np.abs(self.data_y2), 2)) / sigma ** 2
        res += 4 * len(self.data_y) * np.log(sigma)

        return res

    def least_squares_template(self, alpha, phase):
        """
        least squares for minimization - any given template
        :param phase: phase of the template
        :param alpha: scale factor
        :return: cost function - sum of squares
        """
        func_t = alpha * np.array(self.template)  # function to minimize
        func_t = np.roll(func_t, int(phase))

        res = sum(np.power(np.abs(self.data_y - func_t), 2))
        return res

    def least_squares_template2(self, alpha, phase):
        """
        least squares for minimization - 2 templates for 2 datasets with shared phase and scale
        :param alpha: scale factor
        :return: cost function - sum of squares
        """
        func_t = alpha * np.array(self.template)  # function to minimize
        func_t = np.roll(func_t, int(phase))

        func_t2 = alpha * np.array(self.template2)  # function to minimize
        func_t2 = np.roll(func_t2, int(phase))

        res = sum(np.power(np.abs(self.data_y - func_t), 2))
        res2 = sum(np.power(np.abs(self.data_y2 - func_t2), 2))

        return res + res2

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

    def least_squares_bimodal_sine(self, A, A2, f, phi, phi2):
        """
        least squares for minimization - sine function
        :param phi2: phase second sine
        :param A2: amplitude of second sine
        :param A: Amplitude
        :param f: frequency
        :param phi: phase
        :return: cost function - sum of squares
        """
        func_t = A * np.sin(2 * np.pi * f * self.data_x + phi) + A2 * np.sin(
            2 * np.pi * f * self.data_x + phi2)  # function to minimize
        res = sum(np.power(np.abs(self.data_y - func_t), 2))
        return res / self.noise_sigma ** 2

    def least_squares_2sines(self, A, A2, f, f2, phi, delta_phi):
        """
        least squares for minimization - sine function for two datasets
        :param delta_phi: phase difference between two signals
        :param A, A2: Amplitudes of two signals
        :param f, f2: frequencies of two signals
        :param phi: phase
        :return: cost function - sum of squares
        """
        func_t = A * np.sin(2 * np.pi * f * self.data_x + phi)  # function to minimize
        res = sum(np.power(np.abs(self.data_y - func_t), 2)) / self.noise_sigma ** 2

        func_t2 = A * A2 * np.sin(2 * np.pi * f2 * self.data_x + phi + delta_phi)  # function to minimize
        res2 = sum(np.power(np.abs(self.data_y2 - func_t2), 2)) / self.noise_sigma2 ** 2
        return res + res2

    def least_squares_multi_harmonics(self, A, phi, sigma):
        """
        least squares for minimization - sine function for multi datasets
        :param A: Amplitudes of the first harmoincs in the list data_y
        :param f: frequency of the fitted sine
        :param phi: global phase shift
        :return: cost function - sum of squares
        """

        res = 0
        N = len(self.data_y[0])
        for A_, phi_, f_, noise_, data_ in zip(self.harmoincs_amp, self.harmoincs_phases, self.harmoincs_freqs,
                                               self.harmoincs_noise, self.data_y):
            func_t = A * A_ * np.sin(2 * np.pi * f_ * self.data_x + phi_ + phi)  # function to minimize
            res += sum(np.power(np.abs(data_ - func_t), 2)) / noise_
        # print('A = ', A, 'phi = ', phi, 'res = ', res/1e6)
        res /= (sigma ** 2)
        res += 2 * np.log(sigma) * len(self.harmoincs_amp)
        return res

    def least_squares_multi_harmonics2(self, A):
        """
        least squares for minimization - sine function for multi datasets - not log likelihood
        This chi2 is used for finding MLE of gravity+EDM backgrounds
        :param A: Amplitudes of the first harmoincs in the list data_y
        :return: cost function - sum of squares
        """

        res = 0
        N = len(self.data_y[0])
        for A_, phi_, f_, noise_, data_ in zip(self.harmoincs_amp, self.harmoincs_phases, self.harmoincs_freqs,
                                               self.harmoincs_noise, self.data_y):
            func_t = A * A_ * np.sin(2 * np.pi * f_ * self.data_x + phi_)  # function to minimize
            res += sum(np.power(np.abs(data_ - func_t), 2)) / noise_

        return res

    def least_squares_multi_harmonics3(self, A, A2, A3, A4, sigma):
        """
        least squares for minimization - sine function for multi datasets - not log likelihood
        This chi2 is used for finding MLE of gravity+EDM backgrounds
        :param A: Amplitudes of  edm
        :param A2: Amplitudes of edm2
        :return: cost function - sum of squares
        """

        res = 0
        N = len(self.data_y[0])
        for A_, phi_, A2_, phi2_, A3_, phi3_, A4_, phi4_, f_, noise_, data_ in zip(self.harmoincs_amp, self.harmoincs_phases,
                                                                       self.harmoincs_amp2, self.harmoincs_phases2,
                                                                       self.harmoincs_amp3, self.harmoincs_phases3,
                                                                       self.harmoincs_amp4, self.harmoincs_phases4,
                                                                       self.harmoincs_freqs,
                                                                       self.harmoincs_noise, self.data_y):
            func_t = A * A_ * np.sin(2 * np.pi * f_ * self.data_x + phi_) + \
                     A2 * A2_ * np.sin(2 * np.pi * f_ * self.data_x + phi2_) + \
                     A3 * A3_ * np.sin(2 * np.pi * f_ * self.data_x + phi3_) + \
                     A4 * A4_ * np.sin(2 * np.pi * f_ * self.data_x + phi4_)

            res += sum(np.power(np.abs(data_ - func_t), 2)) / noise_

        res /= (sigma ** 2)
        res += 2 * np.log(sigma) * len(self.harmoincs_amp)

        return res

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
        res /= sigma ** 2
        res += 2 * np.log(sigma)

        return res

    def find_mle_template(self, x, template, center_freq, bandwidth, **kwargs):
        """
        The function is fitting the data with a  template using iminuit.
        The fitting is done after applying a bandpass filter to both the template and the data.
        :param template: template for the fit
        :param bandwidth: bandwidth for butter filter [Hz]
        :param center_freq: center frequency for the bandpass filter
        :param x: 1 dim. position data (time domain)
        :return: minimizer result
        """
        # filtering the template and the data
        b, a = signal.butter(3, [2. * (center_freq - bandwidth / 2.) / self.fsamp,
                                 2. * (center_freq + bandwidth / 2.) / self.fsamp], btype='bandpass')
        self.data_y = signal.filtfilt(b, a, x)[5000:-5000]
        self.template = signal.filtfilt(b, a, template)[5000:-5000]

        mimuit_minimizer = Minuit(self.least_squares_template, **kwargs)
        mimuit_minimizer.migrad(ncall=50000)
        return mimuit_minimizer

    def find_mle_PL(self, x, template, scale, center_freq, noise_freq, bandwidth, decimate=10, **kwargs):
        """
        The function is fitting the data with a template using iminuit and the likelihood function
        The fitting is done after applying a bandpass filter to both the template and the data.
        :param decimate: decimate data (good for correlated datasets)
        :param template: template for the fit
        :param bandwidth: bandwidth for butter filter [Hz]
        :param center_freq: center frequency for the bandpass filter
        :param noise_freq: noise bandwidth
        :param x: 1 dim. position data (time domain)
        :return: minimizer result
        """
        # filtering the template and the data
        b, a = signal.butter(3, [2. * (center_freq - bandwidth / 2.) / self.fsamp,
                                 2. * (center_freq + bandwidth / 2.) / self.fsamp], btype='bandpass')
        self.data_y = signal.filtfilt(b, a, x)[5000:-5000:decimate]
        self.template = signal.filtfilt(b, a, template)[5000:-5000:decimate] * scale

        b, a = signal.butter(3, [2. * (noise_freq - bandwidth / 2.) / self.fsamp,
                                 2. * (noise_freq + bandwidth / 2.) / self.fsamp], btype='bandpass')
        self.data_y2 = signal.filtfilt(b, a, x)[5000:-5000:decimate]  # x3 data - QPD carrier phase

        mimuit_minimizer = Minuit(self.log_likelihood_template, **kwargs)
        mimuit_minimizer.migrad(ncall=50000)
        return mimuit_minimizer

    def find_mle_multiHarmoincs(self, x, template, scales, signal_freqs, bandwidth, noises, decimate=10, **kwargs):
        """
        The function is fitting the data with a template using iminuit and the likelihood function
        The fitting is for multiple harmonics simultaneously
        :param noises: list with noise term for each harmonic
        :param scales: scale to convert to force units
        :param decimate: decimate data (good for correlated datasets)
        :param template: template of the signal model
        :param bandwidth: bandwidth for butter filter [Hz]
        :param signal_freqs: frequencies of the different harmonics
        :param x: 1 dim. position data (time domain)
        :return: minimizer result
        """
        # filtering the data at the required frequencies
        # apply a bandpass filter to data and store data in the correct place for the minimization
        self.data_x = np.arange(0, len(x)) / self.fsamp
        self.data_x = self.data_x[5000:-5000:decimate]
        self.harmoincs_freqs = signal_freqs
        self.harmoincs_noise = noises
        self.data_y = []
        for center_freq in signal_freqs:
            b, a = signal.butter(3, [2. * (center_freq - bandwidth / 2.) / self.fsamp,
                                     2. * (center_freq + bandwidth / 2.) / self.fsamp], btype='bandpass')
            self.data_y.append(signal.filtfilt(b, a, x)[5000:-5000:decimate])

        if len(template) == 5000:
            freq = np.fft.rfftfreq(len(template), 1 / self.fsamp)
            fft = np.abs(np.fft.rfft(template)) * 2 / np.sqrt(5000 * 5000)
            angles = (np.angle(np.fft.rfft(template)) + np.pi / 2) % (2 * np.pi)
        else:
            print('Template has to be one second long')

        self.harmoincs_amp = np.array([fft[freq == freq_] * scale_ for freq_, scale_ in zip(signal_freqs, scales)])
        self.harmoincs_phases = np.array([angles[freq == freq_] for freq_ in signal_freqs])

        mimuit_minimizer = Minuit(self.least_squares_multi_harmonics, **kwargs)
        mimuit_minimizer.migrad(ncall=50000)

        return mimuit_minimizer

    def find_mle_multiHarmoincs2(self, x, template, scales, phases, signal_freqs, bandwidth, noises, decimate=10,
                                 **kwargs):
        """
        The function is fitting the data with a template using iminuit and the likelihood function
        The fitting is for multiple harmonics simultaneously
        Phase response added
        :param noises: list with noise term for each harmonic
        :param scales: scale to convert to force units
        :param decimate: decimate data (good for correlated datasets)
        :param template: template of the signal model
        :param bandwidth: bandwidth for butter filter [Hz]
        :param signal_freqs: frequencies of the different harmonics
        :param x: 1 dim. position data (time domain)
        :return: minimizer result
        """
        # filtering the data at the required frequencies
        # apply a bandpass filter to data and store data in the correct place for the minimization
        self.data_x = np.arange(0, len(x)) / self.fsamp
        self.data_x = self.data_x[5000:-5000:decimate]
        self.harmoincs_freqs = signal_freqs
        self.harmoincs_noise = noises
        self.harmoincs_phases = phases
        self.data_y = []
        for center_freq in signal_freqs:
            b, a = signal.butter(3, [2. * (center_freq - bandwidth / 2.) / self.fsamp,
                                     2. * (center_freq + bandwidth / 2.) / self.fsamp], btype='bandpass')
            self.data_y.append(signal.filtfilt(b, a, x)[5000:-5000:decimate])

        if len(template) == 5000:
            freq = np.fft.rfftfreq(len(template), 1 / self.fsamp)
            fft = np.abs(np.fft.rfft(template)) * 2 / np.sqrt(5000 * 5000)
        else:
            print('Template has to be one second long')

        self.harmoincs_amp = np.array([fft[freq == freq_] * scale_ for freq_, scale_ in zip(signal_freqs, scales)])

        mimuit_minimizer = Minuit(self.least_squares_multi_harmonics, **kwargs)
        mimuit_minimizer.migrad(ncall=50000)

        return mimuit_minimizer

    def find_edm_multiHarmoincs(self, x, amps1, phases1, scales, signal_freqs, bandwidth, noises, decimate=10,
                                **kwargs):
        """
        The function is fitting the data with a template of the edm
        The fitting is for multiple harmonics simultaneously
        Phase response added
        :param noises: list with the variance of each harmonic
        :param scales: scale to convert to force units
        :param decimate: decimate data (good for correlated datasets)
        :param amps1: amplitudes of the edm model
        :param phases1: phases of the edm model
        :param bandwidth: bandwidth for butter filter [Hz]
        :param signal_freqs: frequencies of the different harmonics
        :param x: 1 dim. position data (time domain)
        :return: minimizer result
        """
        # filtering the data at the required frequencies
        # apply a bandpass filter to data and store data in the correct place for the minimization
        self.data_x = np.arange(0, len(x)) / self.fsamp
        self.data_x = self.data_x[5000:-5000:decimate]
        self.harmoincs_freqs = signal_freqs
        self.harmoincs_noise = noises

        # set edm model parameters
        self.harmoincs_phases = phases1
        self.harmoincs_amp = np.array([amp_ * scale_ for amp_, scale_ in zip(amps1, scales)])

        # filter data
        self.data_y = []
        for center_freq in signal_freqs:
            b, a = signal.butter(3, [2. * (center_freq - bandwidth / 2.) / self.fsamp,
                                     2. * (center_freq + bandwidth / 2.) / self.fsamp], btype='bandpass')
            self.data_y.append(signal.filtfilt(b, a, x)[5000:-5000:decimate])

        mimuit_minimizer = Minuit(self.least_squares_multi_harmonics2, **kwargs)
        mimuit_minimizer.migrad(ncall=50000)

        return mimuit_minimizer

    def find_edm_multiHarmoincs2(self, x, amps1, phases1, amps2, phases2, amps3, phases3, amps4, phases4, scales, signal_freqs,
                                 bandwidth, noises, decimate=10, **kwargs):
        """
        The function is fitting the data with a template of the edm (3 axis)
        The fitting is for multiple harmonics simultaneously
        Phase response added
        :param noises: list with the variance of each harmonic
        :param scales: scale to convert to force units
        :param decimate: decimate data (good for correlated datasets)
        :param amps1: amplitudes of the edm model
        :param phases1: phases of the edm model
        :param bandwidth: bandwidth for butter filter [Hz]
        :param signal_freqs: frequencies of the different harmonics
        :param x: 1 dim. position data (time domain)
        :return: minimizer result
        """
        # filtering the data at the required frequencies
        # apply a bandpass filter to data and store data in the correct place for the minimization
        self.data_x = np.arange(0, len(x)) / self.fsamp
        self.data_x = self.data_x[5000:-5000:decimate]
        self.harmoincs_freqs = signal_freqs
        self.harmoincs_noise = noises

        # set edm model parameters
        self.harmoincs_phases = phases1
        self.harmoincs_amp = np.array([amp_ * scale_ for amp_, scale_ in zip(amps1, scales)])
        self.harmoincs_phases2 = phases2
        self.harmoincs_amp2 = np.array([amp_ * scale_ for amp_, scale_ in zip(amps2, scales)])
        self.harmoincs_phases3 = phases3
        self.harmoincs_amp3 = np.array([amp_ * scale_ for amp_, scale_ in zip(amps3, scales)])
        self.harmoincs_phases4 = phases4
        self.harmoincs_amp4 = np.array([amp_ * scale_ for amp_, scale_ in zip(amps4, scales)])

        # filter data
        self.data_y = []
        for center_freq in signal_freqs:
            b, a = signal.butter(3, [2. * (center_freq - bandwidth / 2.) / self.fsamp,
                                     2. * (center_freq + bandwidth / 2.) / self.fsamp], btype='bandpass')
            self.data_y.append(signal.filtfilt(b, a, x)[5000:-5000:decimate])

        mimuit_minimizer = Minuit(self.least_squares_multi_harmonics3, **kwargs)
        mimuit_minimizer.migrad(ncall=50000)

        return mimuit_minimizer

    def find_mle_multiHarmoincs_sideband(self, x, template, scales, signal_freqs, bandwidth, noises, decimate=10,
                                         **kwargs):
        """
        The function is fitting the data with a template using iminuit and the likelihood function
        The fitting is for multiple harmonics simultaneously
        :param noises: list with noise term for each harmonic
        :param scales: scale to convert to force units
        :param decimate: decimate data (good for correlated datasets)
        :param template: template of the signal model
        :param bandwidth: bandwidth for butter filter [Hz]
        :param signal_freqs: frequencies of the different harmonics
        :param x: 1 dim. position data (time domain)
        :return: minimizer result
        """
        # filtering the data at the required frequencies
        # apply a bandpass filter to data and store data in the correct place for the minimization
        self.data_x = np.arange(0, len(x)) / self.fsamp
        self.data_x = self.data_x[5000:-5000:decimate]
        self.harmoincs_freqs = signal_freqs
        self.harmoincs_noise = noises
        self.data_y = []
        for center_freq in signal_freqs:
            b, a = signal.butter(3, [2. * ((center_freq + 1) - bandwidth / 2.) / self.fsamp,
                                     2. * ((center_freq + 1) + bandwidth / 2.) / self.fsamp], btype='bandpass')
            self.data_y.append(signal.filtfilt(b, a, x)[5000:-5000:decimate])

        if len(template) == 5000:
            freq = np.fft.rfftfreq(len(template), 1 / self.fsamp)
            fft = np.abs(np.fft.rfft(template)) * 2 / np.sqrt(5000 * 5000)
            angles = (np.angle(np.fft.rfft(template)) + np.pi / 2) % (2 * np.pi)
        else:
            print('Template has to be one second long')

        self.harmoincs_amp = np.array([fft[freq == freq_] * scale_ for freq_, scale_ in zip(signal_freqs, scales)])
        self.harmoincs_phases = np.array([angles[freq == freq_] for freq_ in signal_freqs])

        mimuit_minimizer = Minuit(self.least_squares_multi_harmonics, **kwargs)
        mimuit_minimizer.migrad(ncall=50000)

        return mimuit_minimizer

    def get_PL_multiHarmonics(self, A_array, **kwargs):
        """
        This function must be called after calling find_mle_multiHarmonics, and uses the template and scales and data
        in memory without redefining it.
        :return: array of PL points
        """
        PLarray = []
        kwargs['fix_A'] = True
        for A_ in A_array:
            kwargs['A'] = A_
            mimuit_minimizer = Minuit(self.least_squares_multi_harmonics, **kwargs)
            mimuit_minimizer.migrad(ncall=50000)
            PLarray.append(mimuit_minimizer.fval)
        return np.array(PLarray)

    def get_PL_edm_multiHarmonics(self, A_array, **kwargs):
        """
        This function must be called after calling find_mle_multiHarmonics, and uses the template and scales and data
        in memory without redefining it.
        :return: array of PL points
        """
        PLarray = []
        kwargs['fix_A4'] = True
        for A_ in A_array:
            kwargs['A4'] = A_
            mimuit_minimizer = Minuit(self.least_squares_multi_harmonics3, **kwargs)
            mimuit_minimizer.migrad(ncall=50000)
            PLarray.append(mimuit_minimizer.fval)
            kwargs['A'] = mimuit_minimizer.values[0]
            kwargs['A2'] = mimuit_minimizer.values[1]
            kwargs['A3'] = mimuit_minimizer.values[2]
            kwargs['sigma'] = mimuit_minimizer.values[4]
        return np.array(PLarray)

    def get_PL_sin(self, A_array, **kwargs):
        """
        This function must be called after calling find_mle_sin, and uses the template and scales and data
        in memory without redefining it.
        :return: array of PL points
        """
        PLarray = []
        kwargs['fix_A'] = True
        for A_ in A_array:
            kwargs['A'] = A_
            mimuit_minimizer = Minuit(self.least_squares_sine2, **kwargs)
            mimuit_minimizer.migrad(ncall=50000)
            PLarray.append(mimuit_minimizer.fval)
        return np.array(PLarray)

    def find_mle_template2(self, x2, template2, x3, template3, center_freq, bandwidth, decimate, **kwargs):
        """
        The function is fitting the data with a  template using iminuit.
        The fitting is done after applying a bandpass filter to both the template and the data.
        :param template: template for the fit
        :param bandwidth: bandwidth for butter filter [Hz]
        :param center_freq: center frequency for the bandpass filter
        :param x: 1 dim. position data (time domain)
        :return: minimizer result
        """
        # filtering the template and the data
        b, a = signal.butter(3, [2. * (center_freq - bandwidth / 2.) / self.fsamp,
                                 2. * (center_freq + bandwidth / 2.) / self.fsamp], btype='bandpass')
        self.data_y = signal.filtfilt(b, a, x2)[5000:-5000:decimate]  # x2 data - QPD carrier amplitude
        self.template = signal.filtfilt(b, a, template2)[5000:-5000:decimate]  # x2 template

        self.data_y2 = signal.filtfilt(b, a, x3)[5000:-5000:decimate]  # x3 data - QPD carrier phase
        self.template2 = signal.filtfilt(b, a, template3)[5000:-5000:decimate]  # x3 template

        mimuit_minimizer = Minuit(self.least_squares_template2, **kwargs)
        mimuit_minimizer.migrad(ncall=50000)

        return mimuit_minimizer

    def find_mle_sin(self, x, drive_freq=0, fsamp=5000, bandwidth=50, noise_rms=0, decimate=1, plot=False,
                     suppress_print=True,
                     bimodal=False, **kwargs):
        """
        The function is fitting the data with a sine template using iminuit.
        The fitting is done after applying a bandpass filter.
        :param bimodal: bimodal sine function
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

        if noise_rms != 0:
            self.noise_sigma = noise_rms

        # apply a bandpass filter to data and store data in the correct place for the minimization
        self.data_x = np.arange(0, len(x))[::decimate] / fsamp
        start = time.time()
        if drive_freq != 0:
            if not suppress_print:
                print('Bandpass filter ON. Bandwidth: ', bandwidth, 'Hz')
            b, a = signal.butter(3, [2. * (drive_freq - bandwidth / 2.) / fsamp,
                                     2. * (drive_freq + bandwidth / 2.) / fsamp], btype='bandpass')
            self.data_y = signal.filtfilt(b, a, x)[::decimate]
        else:
            self.data_y = x
        end = time.time()
        if not suppress_print:
            print('bandpass time: ', end - start)

        # we create an instance of Minuit and pass the function to minimize
        if bimodal:
            mimuit_minimizer = Minuit(self.least_squares_bimodal_sine, **kwargs)
        else:
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

    def find_mle_2sin(self, x, x2, drive_freq=0, fsamp=5000, bandwidth=50, noise_rms=0, noise_rms2=0, plot=False,
                      suppress_print=True,
                      **kwargs):
        """
        The function is fitting the data with a sine template using iminuit.
        The fitting is done after applying a bandpass filter.
        :param suppress_print: suppress all printing
        :param noise_rms, noise_rms2: std of the white gaussian noise
        :param plot: plot the data and its fft
        :param bandwidth: bandwidth for butter filter [Hz]
        :param fsamp: sampling rate [1/sec]
        :param x, x2: two 1-dim. position datasets (time domain)
        :param drive_freq: drive frequency of the response
        :return: estimated values, chi_square
        """
        if not suppress_print:
            print('Data overall time: ', len(x) / fsamp, ' sec.')

        if noise_rms != 0:
            self.noise_sigma = noise_rms
            self.noise_sigma2 = noise_rms2

        # apply a bandpass filter to data and store data in the correct place for the minimization
        self.data_x = np.arange(0, len(x)) / fsamp
        start = time.time()
        if drive_freq != 0:
            if not suppress_print:
                print('Bandpass filter ON. Bandwidth: ', bandwidth, 'Hz')
            b, a = signal.butter(3, [2. * (drive_freq - bandwidth / 2.) / fsamp,
                                     2. * (drive_freq + bandwidth / 2.) / fsamp], btype='bandpass')
            self.data_y = signal.filtfilt(b, a, x)
            self.data_y2 = signal.filtfilt(b, a, x2)
        else:
            self.data_y = x
            self.data_y2 = x2
        end = time.time()
        if not suppress_print:
            print('bandpass time: ', end - start)

        # we create an instance of Minuit and pass the function to minimize
        mimuit_minimizer = Minuit(self.least_squares_2sines, **kwargs)

        start = time.time()
        mimuit_minimizer.migrad(ncall=50000)
        end = time.time()
        if not suppress_print:
            print('minimization time: ', end - start)
            print(mimuit_minimizer.get_param_states())

        if plot:
            _, ax = plt.subplots(1, 2, figsize=(9.5, 4))
            # ax[0].scatter(self.data_x, self.data_y)
            # fft = np.abs(np.fft.rfft(x)) ** 2
            # freq = np.fft.rfftfreq(len(x), d=1. / fsamp)
            # ax[0].set(title='raw data')
            plt.subplot(121)
            mimuit_minimizer.draw_profile('A', subtract_min=True)
            plt.subplot(122)
            mimuit_minimizer.draw_profile('A2', subtract_min=True)
            plt.show()
        if not suppress_print:
            print('reduced chi2: ', mimuit_minimizer.fval / (2 * len(self.data_y) - 4))

        return mimuit_minimizer
