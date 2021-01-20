import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal

from likelihood_calculator import likelihood_analyser


class DMAnalyser:
    def __init__(self):
        self.BDFs = None  # a list of BeadDataFiles
        self.minimizer_1d_results = None  # using x2 only
        self.minimizer_2d_results = None  # using x2 and x3
        self.noise_rms_x2 = 1  # x2 noise gaussian width
        self.noise_rms_x3 = 1  # x3 noise gaussian width
        self.noise_list_x2 = []  # x2 noise values per  BDF
        self.noise_list_x3 = []  # x3 noise values per BDF
        self.avg_list_x2 = []  # x2 average response
        self.avg_list_x3 = []  # x3 average response

    def estimate_noise(self, bandwidth=10, frequency=151):
        """
        Estimation of the noise level outside of the signal-band
        :param frequency: The frequency of the carrier signal
        :type bandwidth: bandpass filter bandpass
        """
        for i, bb in enumerate(self.BDFs):
            xx3 = bb.response_at_freq3('x', frequency, bandwidth=bandwidth) / 6
            analytic_signal3 = signal.hilbert(xx3)
            amplitude_envelope3 = np.abs(analytic_signal3)
            self.noise_list_x3.append(np.std(amplitude_envelope3[5000:-5000]))

            xx2 = bb.response_at_freq2('x', frequency, bandwidth=bandwidth) * 50000
            analytic_signal2 = signal.hilbert(xx2)
            amplitude_envelope2 = np.abs(analytic_signal2)
            self.noise_list_x2.append(np.std(amplitude_envelope2[5000:-5000]))
        self.noise_rms_x2 = np.mean(self.noise_list_x2)
        self.noise_rms_x3 = np.mean(self.noise_list_x3)
        print('x2 noise rms: ', self.noise_rms_x2)
        print('x3 noise rms: ', self.noise_rms_x3)

    def plot_dataset(self, bdf_i, bandwidth=10, frequency=151):
        """
        Plot the x2 and x3 data and their envelopes
        :param frequency: carrier frequency
        :param bandwidth: carrier frequency bandpass bandwidth
        :param bdf_i: index of BDF to be shown
        """
        bb = self.BDFs[bdf_i]
        xx = bb.response_at_freq2('x', frequency, bandwidth=bandwidth) * 50000
        analytic_signal = signal.hilbert(xx)
        amplitude_envelope = np.abs(analytic_signal)

        xx3 = bb.response_at_freq3('x', frequency, bandwidth=bandwidth) / 6
        analytic_signal3 = signal.hilbert(xx3)
        amplitude_envelope3 = np.abs(analytic_signal3)

        _, ax = plt.subplots(1, 2, figsize=(9.5, 4))
        t = np.arange(300000) / 5000
        ax[0].plot(t, xx, label='signal')
        ax[0].plot(t, amplitude_envelope, label='envelope')
        ax[0].set(xlabel='time[sec]', title='X2')

        ax[1].plot(t, xx3, label='signal')
        ax[1].plot(t, amplitude_envelope3, label='envelope')
        ax[1].set(xlabel='time[sec]', title='X3')
        plt.show()

    def get_delta_alpha(self, bdf_i, alpha_frequency, bandwidth=10, frequency=151):
        """
        :param alpha_frequency: frequency of the fine constant oscillations
        :param frequency: carrier frequency
        :param bandwidth: carrier frequency bandpass bandwidth
        :param bdf_i: index of the bdf dataset to be used
        :return: mean_delta_alpha, iminuit minimizer
        """
        bb = self.BDFs[bdf_i]
        xx3 = bb.response_at_freq3('x', frequency, bandwidth=bandwidth) / 6
        analytic_signal3 = signal.hilbert(xx3)
        amplitude_envelope3 = np.abs(analytic_signal3)
        average3 = np.mean(amplitude_envelope3[5000:-5000])
        envelope3_subtracted = amplitude_envelope3 - average3
        envelope3_subtracted = envelope3_subtracted[5000:-5000]

        xx2 = bb.response_at_freq2('x', frequency, bandwidth=bandwidth) * 50000
        analytic_signal2 = signal.hilbert(xx2)
        amplitude_envelope2 = np.abs(analytic_signal2)
        average2 = np.mean(amplitude_envelope2[5000:-5000])
        envelope2_subtracted = amplitude_envelope2 - average2
        envelope2_subtracted = envelope2_subtracted[5000:-5000]

        fit_kwargs = {'A': 0, 'f': alpha_frequency, 'phi': 0, 'A2': average2 / average3, 'f2': alpha_frequency,
                      'delta_phi': 0,
                      'error_A': 0.01, 'error_f': 1, 'error_phi': 0.1, 'errordef': 1,
                      'error_A2': 2, 'error_f2': 10, 'error_delta_phi': 0.1,
                      'limit_phi': [0, 2 * np.pi], 'limit_delta_phi': [-0.1, 0.1],
                      'limit_A': [-1, 1],
                      'print_level': 0, 'fix_f': True, 'fix_phi': False, 'fix_f2': True, 'fix_delta_phi': True,
                      'fix_A2': True}
        lc_i = likelihood_analyser.LikelihoodAnalyser()
        if alpha_frequency < 1:
            fsamp = 50
            step = 100
        else:
            fsamp = 100
            step = 50

        m1_tmp = lc_i.find_mle_2sin(envelope3_subtracted[::step], envelope2_subtracted[::step], fsamp=fsamp,
                                    noise_rms=self.noise_list_x3[bdf_i],
                                    noise_rms2=self.noise_list_x2[bdf_i],
                                    plot=False, suppress_print=True, **fit_kwargs)
        delta_alpha = m1_tmp.values[0] / average3

        print('***************************************************')
        print('bdf_i: ', bdf_i, ', AM frequency: ', alpha_frequency)
        print('sensitivity: ', '{:.2e}'.format(np.abs(delta_alpha)))

        return np.abs(delta_alpha), m1_tmp

    def get_sensitivity(self, alpha_frequency, bandwidth=10, frequency=151):
        """
        Calculating the sensitivity at a specific frequency using all the datasets
        :param alpha_frequency: DM frequency
        :param bandwidth: carrier bandpass bandwidth
        :param frequency: carrier frequency
        :return: sensitivity on delta_alpha
        """
        delta_alpha_list = []
        self.minimizer_2d_results = []
        for i, bb in enumerate(self.BDFs):
            alpha_tmp, m1 = self.get_delta_alpha(bdf_i=i, alpha_frequency=alpha_frequency, bandwidth=bandwidth,
                                                 frequency=frequency)
            self.minimizer_2d_results.append(m1)
            delta_alpha_list.append(alpha_tmp)

        print('***************************************************')
        print('average: ', np.mean(delta_alpha_list), 'std: ', np.std(delta_alpha_list))
        print('standard error: ', np.std(delta_alpha_list) / np.sqrt(len(delta_alpha_list)))

        return np.std(delta_alpha_list) / np.sqrt(len(delta_alpha_list))
