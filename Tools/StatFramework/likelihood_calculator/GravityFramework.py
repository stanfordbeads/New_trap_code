import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal

from likelihood_calculator import likelihood_analyser
from likelihood_calculator import auxiliary_functions as aux

import sys

sys.path.append('/home/analysis_user/New_trap_code/SensitivityFramework/')
from signal_model_utilities import *


class GravityFramework:
    def __init__(self):
        self.BDFs = None  # a list of BeadDataFiles
        self.minimizer_1d_results = None  # using x2 only
        self.minimizer_2d_results = None  # using x2 and x3
        self.noise_rms_x2 = 1  # x2 noise gaussian width
        self.noise_rms_x3 = 1  # x3 noise gaussian width
        self.noise_rms_z2 = 1  # z2 noise gaussian width
        self.noise_list_x2 = []  # x2 noise values per BDF - sideband
        self.noise_list_x3 = []  # x3 noise values per BDF - sideband
        self.noise_list_z2 = []  # z2 noise values per BDF - sideband
        self.avg_list_x2 = []  # x2 average response - force calibration files
        self.avg_list_x3 = []  # x3 average response - force calibration files
        self.fundamental_freq = 13  # fundamental frequency
        self.Harmonics_list = None  # list of frequencies
        self.Harmonics_array = None  # amplitudes at the given harmonics
        self.Error_array = None  # errors of the amplitudes
        self.scale_X2 = 1  # scale X2 signal to force in Newtons
        self.scale_X3 = 1  # scale X3 signal to force in Newtons
        self.scale_Y2 = 1  # scale X2 signal to force in Newtons
        self.scale_Z2 = 1  # scale Z2 signal to force in Newtons
        self.A2_mean = 1  # X3/X2 mean
        self.fsamp = 5000
        self.lc_i = likelihood_analyser.LikelihoodAnalyser()
        self.m1_list = None  # last m1 list (last fitting)
        self.last_phase = 0
        self.last_A = 0
        self.tf_ffts = None
        self.tf_freq = None

    def plot_dataset(self, bdf_i, res=50000):
        """
        Plot the x2 and x3 data and their envelopes
        :param res: resolution of the fft
        :param bdf_i: index of BDF to be shown
        """

        bdf = self.BDFs[bdf_i]
        x2_psd, freqs = matplotlib.mlab.psd(bdf.x2 * 50000, Fs=self.fsamp, NFFT=res, detrend='default')
        x3_psd, _ = matplotlib.mlab.psd(bdf.x3 / 6, Fs=self.fsamp, NFFT=res, detrend='default')

        _, ax = plt.subplots(1, 2, figsize=(9.5, 4))
        ax[0].loglog(freqs, x2_psd)
        ax[1].loglog(freqs, x3_psd)

        plt.show()

    def get_amplitude(self, bdf, noise_rms, noise_rms2, bandwidth=1, decimate=10, **fit_kwargs):
        """
        Fit and extract the amplitude of one harmonic from one particular file
        :param bandwidth: bandpass bandwidth
        :param noise_rms, noise_rms2: noise std of X2 and X3
        :param bdf: bdf dataset to be used
        :return: amplitude, error
        """
        bb = bdf
        frequency = fit_kwargs['f']

        xx2 = bb.response_at_freq2('x', frequency, bandwidth=bandwidth) * 50000
        xx2 = xx2[5000:-5000:decimate]  # cut out the first and last second

        xx3 = bb.response_at_freq3('x', frequency, bandwidth=bandwidth) / 6
        xx3 = xx3[5000:-5000:decimate]  # cut out the first and last second

        m1_tmp = self.lc_i.find_mle_2sin(xx2, xx3, fsamp=self.fsamp / decimate,
                                         noise_rms=noise_rms,
                                         noise_rms2=noise_rms2,
                                         plot=False, suppress_print=True, **fit_kwargs)

        print('***************************************************')
        print('X2-amplitude: ', '{:.2e}'.format(np.abs(m1_tmp.values[0])))
        print('reduced chi2: ', m1_tmp.fval / (len(xx2) - 3))

        return m1_tmp.values[0], m1_tmp.errors[0], m1_tmp

    def get_z_amplitude(self, bdf, noise_rms, bandwidth=1, decimate=10, bimodal=False, **fit_kwargs):
        """
        Fit and extract the Z2 amplitude of one harmonic from one particular file
        :param bimodal: bimodal fit to account for spin
        :param decimate: decimate the data before fitting
        :param bandwidth: bandpass bandwidth
        :param noise_rms: noise std of Z2 and X3
        :param bdf: bdf dataset to be used
        :return: amplitude, error
        """
        bb = bdf
        frequency = fit_kwargs['f']

        xx2 = bb.response_at_freq2('z', frequency, bandwidth=bandwidth)
        xx2 = xx2[5000:-5000:decimate]  # cut out the first and last second

        m1_tmp = self.lc_i.find_mle_sin(xx2, fsamp=self.fsamp / decimate,
                                        noise_rms=noise_rms,
                                        plot=False, suppress_print=True, bimodal=bimodal, **fit_kwargs)

        print('***************************************************')
        print('Z2-amplitude: ', '{:.2e}'.format(np.abs(m1_tmp.values[0])))
        print('reduced chi2: ', m1_tmp.fval / (len(xx2) - 2))

        return m1_tmp.values[0], m1_tmp.errors[0], m1_tmp

    def get_y_amplitude(self, bdf, noise_rms, bandwidth=1, decimate=10, **fit_kwargs):
        """
        Fit and extract the Y2 amplitude of one harmonic from one particular file
        :param decimate: decimate the data before fitting
        :param bandwidth: bandpass bandwidth
        :param noise_rms: noise std of Y2
        :param bdf: bdf dataset to be used
        :return: amplitude, error
        """
        bb = bdf
        frequency = fit_kwargs['f']

        xx2 = bb.response_at_freq2('y', frequency, bandwidth=bandwidth) * 50000
        xx2 = xx2[5000:-5000:decimate]  # cut out the first and last second

        m1_tmp = self.lc_i.find_mle_sin(xx2, fsamp=self.fsamp / decimate,
                                        noise_rms=noise_rms,
                                        plot=False, suppress_print=True, bimodal=False, **fit_kwargs)

        print('***************************************************')
        print('Y2-amplitude: ', '{:.2e}'.format(np.abs(m1_tmp.values[0])))
        print('reduced chi2: ', m1_tmp.fval / (len(xx2) - 2))

        return m1_tmp.values[0], m1_tmp.errors[0], m1_tmp

    def get_alpha(self, bdf, center_freq, bandwidth, direction='x', x_focous=400, frequency=13,
                  lambda_par=100e-6, height=0e-6, suppress_print=True, **fit_kwargs):
        """
         Fit and extract the scale factor for the yukawa force compared to 10^10
         :param bandwidth: bandpass bandwidth
         :param center_freq: bandpass filter center frequency
         :param bdf: bdf dataset to be used
         :param direction: force firection
         :return: amplitude, error
         """
        stroke = np.std(bdf.cant_pos[1] * 50) * np.sqrt(2) * 2  # stroke in y in micrometers
        cant_pos_x = np.mean(bdf.cant_pos[0])  # cantilever position in x for distance to sphere - in micrometers
        separation = x_focous - aux.voltage_to_position(cant_pos_x) - 4.8 / 2
        time_sec = len(bdf.x2) / self.fsamp

        if not suppress_print:
            print('Separation (face to face): ', separation)
            print('Stroke: ', stroke)
            print('Time: ', time_sec)

        template = force_vs_time(separation=separation * 1e-6, height=height, stroke=stroke * 1e-6, frequency=frequency,
                                 direction=direction, lambda_par=lambda_par, yuk_or_grav="yuk", alpha=1e10)
        template = list(template[1]) * int(time_sec)

        if direction == 'x':
            xx = bdf.x2 * 50000
            tmp_scale = self.scale_X2
        elif direction == 'z':
            xx = bdf.z2
            tmp_scale = self.scale_Z2

        m1_tmp = self.lc_i.find_mle_template(xx, np.array(template) * tmp_scale,
                                             center_freq=center_freq,
                                             bandwidth=bandwidth, **fit_kwargs)

        print('***************************************************')
        print('alpha: ', '{:.2e}'.format(m1_tmp.values[0]))
        print('reduced chi2: ', m1_tmp.fval / (len(bdf.x2) - 1))

        return m1_tmp.values[0], m1_tmp.errors[0], m1_tmp

    def get_alpha_2d(self, bdf, center_freq, bandwidth, direction1='x', direction2='z', x_focous=400, frequency=13,
                     lambda_par=100e-6, height=0e-6, decimate=10, suppress_print=True, **fit_kwargs):
        """
         Fit and extract the scale factor for the yukawa force compared to 10^10
         The function is performing the fit using two axes in a correlated way
         :param decimate: decimate data for speedup
         :param bandwidth: bandpass bandwidth
         :param center_freq: bandpass filter center frequency
         :param bdf: bdf dataset to be used
         :param direction1: force direction of first axis - can be 'x','z','x3'
         :param direction2: force direction of second axis - can be 'x','z','x3'
         :return: amplitude, error
         """
        # temporally overriding the stroke and separation parameters - for sensitivity estimation purposes
        stroke = np.std(bdf.cant_pos[1] * 50) * np.sqrt(2) * 2  # stroke in y in micrometers
        cant_pos_x = np.mean(bdf.cant_pos[0])  # cantilever position in x for distance to sphere - in micrometers
        separation = x_focous - aux.voltage_to_position(cant_pos_x) - 4.8 / 2
        # time_sec = len(bdf.x2) / self.fsamp
        # stroke = 100  # in microns
        # separation = 6.5  # in microns

        if not suppress_print:
            print('Separation (face to face): ', separation)
            print('Stroke: ', stroke)
            print('Time: ', time_sec)

        # prepare the two templates for the fit
        if direction1 == 'x3':
            direction_tmp = 'x'
        else:
            direction_tmp = direction1
        template1 = force_vs_time(separation=separation * 1e-6, height=height, stroke=stroke * 1e-6,
                                  frequency=frequency,
                                  direction=direction_tmp, lambda_par=lambda_par, yuk_or_grav="yuk", alpha=1e10)
        if direction2 == 'x3':
            direction_tmp = 'x'
        else:
            direction_tmp = direction2
        template2 = force_vs_time(separation=separation * 1e-6, height=height, stroke=stroke * 1e-6,
                                  frequency=frequency,
                                  direction=direction_tmp, lambda_par=lambda_par, yuk_or_grav="yuk", alpha=1e10)
        template1 = list(template1[1]) * int(time_sec)
        template2 = list(template2[1]) * int(time_sec)

        # data preparation
        if direction1 == 'x':
            xx1 = bdf.x2 * 50000
            tmp_scale1 = self.scale_X2 * np.interp(center_freq, self.tf_freq, self.tf_ffts[0])
        elif direction1 == 'x3':
            xx1 = bdf.x3 / 6
            tmp_scale1 = self.scale_X3 * np.interp(center_freq, self.tf_freq, self.tf_ffts[0])
        elif direction1 == 'z':
            xx1 = bdf.z2
            tmp_scale1 = self.scale_Z2 * np.interp(center_freq, self.tf_freq, self.tf_ffts[2])

        if direction2 == 'x':
            xx2 = bdf.x2 * 50000
            tmp_scale2 = self.scale_X2 ** np.interp(center_freq, self.tf_freq, self.tf_ffts[0])
        elif direction1 == 'x3':
            xx2 = bdf.x3 / 6
            tmp_scale2 = self.scale_X3 * np.interp(center_freq, self.tf_freq, self.tf_ffts[0])
        elif direction2 == 'z':
            xx2 = bdf.z2
            tmp_scale2 = self.scale_Z2 * np.interp(center_freq, self.tf_freq, self.tf_ffts[2])

        # find the mle
        m1_tmp = self.lc_i.find_mle_template2(xx1, np.array(template1) * tmp_scale1,
                                              xx2, np.array(template2) * tmp_scale2,
                                              center_freq=center_freq, bandwidth=bandwidth, decimate=decimate,
                                              **fit_kwargs)

        print('***************************************************')
        print('alpha: ', '{:.2e}'.format(m1_tmp.values[0]))
        print('reduced chi2: ', m1_tmp.fval / (len(bdf.x2) - 1))

        return m1_tmp.values[0], m1_tmp.errors[0], m1_tmp

    def get_alpha2(self, bdf, center_freq, bandwidth, x_focous=400, frequency=13,
                   lambda_par=100e-6, height=0e-6, suppress_print=True, **fit_kwargs):
        """
         Fit and extract the scale factor for the yukawa force compared to 10^10
         This function uses the correlated X2 and X3 signals - QPD signal amplitude and phase
         :param bandwidth: bandpass bandwidth
         :param center_freq: bandpass filter center frequency
         :param bdf: bdf dataset to be used
         :return: amplitude, error
         """
        stroke = np.std(bdf.cant_pos[1] * 50) * np.sqrt(2) * 2  # stroke in y in micrometers
        cant_pos_x = np.mean(bdf.cant_pos[0])  # cantilever position in x for distance to sphere - in volts
        separation = x_focous - aux.voltage_to_position(cant_pos_x) - 4.8 / 2
        time_sec = len(bdf.x2) / self.fsamp

        if not suppress_print:
            print('Separation (face to face): ', separation)
            print('Stroke: ', stroke)
            print('Time: ', time_sec)

        template = force_vs_time(separation=separation * 1e-6, height=height, stroke=stroke * 1e-6, frequency=frequency,
                                 direction="x", lambda_par=lambda_par, yuk_or_grav="yuk", alpha=1e10)
        template = list(template[1]) * int(time_sec)

        m1_tmp = self.lc_i.find_mle_template2(x2=bdf.x2 * 50000, template2=np.array(template) * self.scale_X2,
                                              x3=bdf.x3 / 6, template3=np.array(template) * self.scale_X3,
                                              center_freq=center_freq, bandwidth=bandwidth, **fit_kwargs)

        print('***************************************************')
        print('alpha: ', '{:.2e}'.format(m1_tmp.values[0]))
        print('reduced chi2: ', m1_tmp.fval / (len(bdf.x2) - 2))

        return m1_tmp.values[0], m1_tmp.errors[0], m1_tmp

    def build_noise_array(self, sideband_freq, bandwidth=1):
        self.noise_list_x2 = []
        self.noise_list_x3 = []

        for bb in self.BDFs:
            xx2 = bb.response_at_freq2('x', sideband_freq, bandwidth=bandwidth) * 50000
            self.noise_list_x2.append(np.std(xx2[5000:-5000]))

            xx3 = bb.response_at_freq3('x', sideband_freq, bandwidth=bandwidth) / 6
            self.noise_list_x3.append(np.std(xx3[5000:-5000]))

        self.noise_rms_x2 = np.mean(self.noise_list_x2)
        self.noise_rms_x3 = np.mean(self.noise_list_x3)
        print('x2 noise rms: ', self.noise_rms_x2)
        print('x3 noise rms: ', self.noise_rms_x3)

    def build_noise_array_z(self, sideband_freq, bandwidth=1):
        self.noise_list_z2 = []

        for bb in self.BDFs:
            xx2 = bb.response_at_freq2('z', sideband_freq, bandwidth=bandwidth)
            self.noise_list_z2.append(np.std(xx2[5000:-5000]))

        self.noise_rms_z2 = np.mean(self.noise_list_z2)
        print('z2 noise level: ', self.noise_rms_z2, ' std: ', np.std(self.noise_list_z2))

    def build_x_response(self, bdf_list, drive_freq, charges, bandwidth, decimate=10):
        """
        Calculates the X response by fitting X2 and X3 simultaneously
        :param bdf_list: list of force calibration BeadDataFiles
        :param drive_freq: the drive frequency on the electrodes
        :param charges: charge state on the sphere
        :return: m1_tmp, list of the minimizer
        """
        fit_kwargs = {'A': 10, 'f': drive_freq, 'phi': 0, 'A2': 2, 'f2': drive_freq,
                      'delta_phi': 0,
                      'error_A': 1, 'error_f': 1, 'error_phi': 0.5, 'errordef': 1,
                      'error_A2': 1, 'error_f2': 1, 'error_delta_phi': 0.1,
                      'limit_phi': [-2 * np.pi, 2 * np.pi], 'limit_delta_phi': [-2 * np.pi, 2 * np.pi],
                      'limit_A': [0, 1000], 'limit_A2': [0, 1000],
                      'print_level': 0, 'fix_f': True, 'fix_phi': False, 'fix_f2': True, 'fix_delta_phi': False,
                      'fix_A2': False}

        m1_tmp = [self.get_amplitude(bdf=bdf_, noise_rms=1, noise_rms2=1, bandwidth=bandwidth, **fit_kwargs)[2] for
                  bdf_ in bdf_list]

        force = charges * 1.6e-19 * 20 / 8e-3 * 0.61  # in Newtons
        A_mean = np.mean([m1.values[0] for m1 in m1_tmp])
        A2_mean = np.mean([m1.values[1] for m1 in m1_tmp])
        self.scale_X2 = A_mean / force
        self.scale_X3 = A_mean * A2_mean / force
        self.A2_mean = A2_mean

        print('X3 to X2 ratio:', A2_mean)
        print('X2 response (amplitude):', A_mean)
        print('X2 response (amplitude):', A_mean)
        self.m1_list = m1_tmp

        return m1_tmp

    def build_z_response(self, bdf_list, drive_freq, charges, bandwidth, decimate=10,
                         include_sigma=False, bimodal=False):
        """
        Calculates the Z response by fitting sine
        :param bimodal: bimodal fit of response to account for spin
        :param decimate: decimate data for speedup
        :param include_sigma: include sigma in the fit
        :param bandwidth: bandwidth for the bandpass filter
        :param bdf_list: list of force calibration BeadDataFiles
        :param drive_freq: the drive frequency on the electrodes
        :param charges: charge state on the sphere
        :return: m1_tmp, list of the minimizer
        """
        fit_kwargs = {'A': -10, 'f': drive_freq, 'phi': 0.3,
                      'error_A': 1, 'error_f': 1, 'error_phi': 0.5, 'errordef': 1,
                      'limit_phi': [-2 * np.pi, 2 * np.pi],
                      'limit_A': [-10000, 10000],
                      'print_level': 0, 'fix_f': True, 'fix_phi': False}
        if bimodal:
            fit_kwargs = {'A': 10, 'f': drive_freq, 'phi': 0.18,
                          'A2': 1, 'phi2': 0.5, 'error_A2': 1,
                          'error_A': 1, 'error_f': 1, 'error_phi': 0.5, 'error_phi2': 0.5, 'errordef': 1,
                          'limit_phi': [-2 * np.pi, 2 * np.pi], 'limit_phi2': [-2 * np.pi, 2 * np.pi],
                          'limit_A': [0, 100000], 'limit_A2': [0, 100000],
                          'print_level': 0, 'fix_f': True, 'fix_phi': False}
        if include_sigma:
            fit_kwargs = {'A': 0, 'f': drive_freq, 'phi': 0,
                          'error_A': 1, 'error_f': 1, 'error_phi': 1, 'errordef': 1,
                          'limit_phi': [-2 * np.pi, 2 * np.pi],
                          'limit_A': [-10000, 100000],
                          'print_level': 0, 'fix_f': True, 'fix_phi': False, 'fix_A': False,
                          'sigma': self.noise_rms_z2, 'fix_sigma': False, 'limit_sigma': [0, None]}

        m1_tmp = [self.get_z_amplitude(bdf=bdf_, noise_rms=1, bandwidth=bandwidth, decimate=decimate,
                                       bimodal=bimodal, **fit_kwargs)[2] for bdf_ in bdf_list]

        force = charges * 1.6e-19 * 20 / 8e-3 * 0.61  # in Newtons
        A_mean = np.mean([m1.values[0] for m1 in m1_tmp])
        self.scale_Z2 = A_mean / force
        print('Z2 response (amplitude):', A_mean)
        self.m1_list = m1_tmp

        return m1_tmp

    def build_y_response(self, bdf_list, drive_freq, charges, bandwidth, decimate=10):
        """
        Calculates the Y response by fitting sine
        :param decimate: decimate data for speedup
        :param bandwidth: bandwidth for the bandpass filter
        :param bdf_list: list of force calibration BeadDataFiles
        :param drive_freq: the drive frequency on the electrodes
        :param charges: charge state on the sphere
        :return: m1_tmp, list of the minimizer
        """
        fit_kwargs = {'A': 10, 'f': drive_freq, 'phi': 0.0,
                      'error_A': 1, 'error_f': 1, 'error_phi': 0.5, 'errordef': 1,
                      'limit_phi': [-2 * np.pi, 2 * np.pi],
                      'limit_A': [0, 100000],
                      'print_level': 0, 'fix_f': True, 'fix_phi': False}

        m1_tmp = [self.get_y_amplitude(bdf=bdf_, noise_rms=1, bandwidth=bandwidth, decimate=decimate,
                                       **fit_kwargs)[2] for bdf_ in bdf_list]

        force = charges * 1.6e-19 * 20 / 8e-3 * 0.61  # in Newtons
        A_mean = np.mean([m1.values[0] for m1 in m1_tmp])
        self.scale_Y2 = A_mean / force
        print('Y2 response (amplitude):', A_mean)
        self.m1_list = m1_tmp

        return m1_tmp

    def build_transfer_function(self, bdf_xyz, base_freq=7, number_of_harmonics=50, scale_freq=151, plot=False):
        """
        Set the transfer function for x, y, and z.
        :param bdf_xyz: a list of x,y,z bdfs
        :param base_freq: base frequency of the transfer function measurement
        :param number_of_harmonics: number of harmonics in the transfer function measurement
        :param scale_freq: the frequency to scale to 1 (should be the force calibration frequency)
        :param plot: plot transfer function datasets
        :return: nothing so far
        """
        self.tf_ffts = []
        axes = ['x', 'y', 'z']
        freq_tmp, _ = bdf_xyz[0].psd2('x')
        freqs_tmp = [freq_tmp == base_freq * i for i in range(1, number_of_harmonics + 1)]
        indices = [i == 1 for i in np.sum(freqs_tmp, axis=0)]
        self.tf_freq = freq_tmp[indices]
        for i, bb in enumerate(bdf_xyz):
            _, fft_tmp = bb.psd2(axes[i])
            fft_tmp = np.sqrt(fft_tmp)[indices]
            scale = np.interp(scale_freq, self.tf_freq, fft_tmp)
            self.tf_ffts += [fft_tmp / scale]

        if plot:
            _, ax = plt.subplots(1, 2, figsize=(9.5, 4))
            [ax[0].loglog(self.tf_freq, fft_, '.') for fft_ in self.tf_ffts[:2]]
            ax[1].loglog(self.tf_freq, self.tf_ffts[2], '.')

    def build_harmonics_array(self, freq):
        """
        Calculate the amplitude for all BDFs at a specific frequency
        :param freq: frequency to be tested
        :return: response (X2 amplitude) array
        """
        m1_tmp = []
        for i, bdf_ in enumerate(self.BDFs):
            print(i, '/', len(self.BDFs))

            fit_kwargs = {'A': self.last_A, 'f': freq, 'phi': self.last_phase, 'A2': self.A2_mean, 'f2': freq,
                          'delta_phi': 0,
                          'error_A': 1, 'error_f': 1, 'error_phi': 0.5, 'errordef': 1,
                          'error_A2': 1, 'error_f2': 1, 'error_delta_phi': 0.1,
                          'limit_phi': [-2 * np.pi, 2 * np.pi], 'limit_delta_phi': [-0.1, 0.1],
                          'limit_A': [0, 1000], 'limit_A2': [0, 1000],
                          'print_level': 0, 'fix_f': True, 'fix_phi': False, 'fix_f2': True, 'fix_delta_phi': True,
                          'fix_A2': True}

            m1_tmp2 = self.get_amplitude(bdf=bdf_, noise_rms=self.noise_list_x2[i], noise_rms2=self.noise_list_x3[i],
                                         **fit_kwargs)[2]
            self.last_A = m1_tmp2.values[0]
            self.last_phase = m1_tmp2.values[1]
            m1_tmp.append(m1_tmp2)

        self.m1_list = m1_tmp
        self.Harmonics_list = [freq]
        self.Harmonics_array = np.array([m1.values[0] for m1 in m1_tmp])

        A_mean = np.mean(self.Harmonics_array)

        print('X [N]:', A_mean / self.scale_X2)
        print('X2 response (amplitude):', A_mean)

        return self.Harmonics_array, m1_tmp

    def get_alpha_mle_pl(self, bdf, center_freq, noise_freq, bandwidth, decimate=10, direction1='x', x_focous=400,
                         frequency=13, offset_y=0,
                         lambda_par=100e-6, height=0e-6, suppress_print=True, large_bead=False, **fit_kwargs):
        """
         Fit and extract the scale factor for the yukawa force compared to 10^8
         The function is performing the fit using two axes in a correlated way
         :param large_bead: set to true if 7.6 um German beads are used (4.8um is used otherwise)
         :param offset_y: y offset of the attractor
         :param lambda_par: lambda parameter for the Yukawa term
         :param frequency: attractor shaking frequency
         :param height: attractor height
         :param decimate: decimate data before the fit
         :param bandwidth: bandpass bandwidth
         :param center_freq: bandpass filter center frequency
         :param noise_freq: noise dataset center frequency
         :param bdf: bdf dataset to be used
         :param direction1: force direction of first axis - can be 'x','z','x3'
         :return: amplitude, error
         """
        # temporally overriding the stroke and separation parameters - for sensitivity estimation purposes
        stroke = np.std(bdf.cant_pos[1] * 50) * np.sqrt(2) * 2  # stroke in y in micrometers
        cant_pos_x = np.mean(bdf.cant_pos[0])  # cantilever position in x for distance to sphere - in micrometers
        if large_bead:
            separation = x_focous - aux.voltage_to_position(cant_pos_x) - 7.6 / 2
        else:
            separation = x_focous - aux.voltage_to_position(cant_pos_x) - 4.8 / 2
        time_sec = len(bdf.x2) / self.fsamp
        # stroke = 100  # in microns
        # separation = 6.5  # in microns

        if not suppress_print:
            print('Large Bead: ', large_bead)
            print('Separation (face to face): ', separation)
            print('Stroke: ', stroke)
            print('Time: ', time_sec)

        # prepare the two templates for the fit
        if direction1 == 'x3':
            direction_tmp = 'x'
        else:
            direction_tmp = direction1

        if large_bead:
            template1 = force_vs_time(separation=separation * 1e-6, height=height, stroke=stroke * 1e-6,
                                      frequency=frequency,
                                      direction=direction_tmp, lambda_par=lambda_par, offset_y=offset_y,
                                      yuk_or_grav="yuk", alpha=1e10, bead_size=3.8e-6)
        else:
            template1 = force_vs_time(separation=separation * 1e-6, height=height, stroke=stroke * 1e-6,
                                      frequency=frequency,
                                      direction=direction_tmp, lambda_par=lambda_par, offset_y=offset_y,
                                      yuk_or_grav="yuk", alpha=1e10)

        template1 = np.array(list(template1[1]) * int(time_sec))

        # data preparation
        if direction1 == 'x':
            xx1 = bdf.x2 * 50000
            tmp_scale1 = self.scale_X2 * np.interp(center_freq, self.tf_freq, self.tf_ffts[0])
        elif direction1 == 'x3':
            xx1 = bdf.x3 / 6
            tmp_scale1 = self.scale_X3 * np.interp(center_freq, self.tf_freq, self.tf_ffts[0])
        elif direction1 == 'z':
            xx1 = bdf.z2
            tmp_scale1 = self.scale_Z2 * np.interp(center_freq, self.tf_freq, self.tf_ffts[2])

        # find the mle
        m1_tmp = self.lc_i.find_mle_PL(xx1, np.array(template1), tmp_scale1,
                                       center_freq=center_freq, noise_freq=noise_freq,
                                       bandwidth=bandwidth, decimate=decimate, **fit_kwargs)

        print('***************************************************')
        print('alpha mle: ', '{:.2e}'.format(m1_tmp.values[0]))
        print('sigma mle: ', '{:.2e}'.format(m1_tmp.values[2]))
        print('reduced chi2: ', m1_tmp.fval / (len(bdf.x2) - 1))

        return m1_tmp.values[0], m1_tmp.errors[0], m1_tmp

    def get_alpha_mle_multiHarmonics(self, bdf, freqs_array, noise_array, bandwidth, decimate=10, x_focous=400,
                                     y_focous=200,
                                     lambda_par=100e-6, height=0e-6, suppress_print=True, large_bead=False,
                                     **fit_kwargs):
        """
         Fit and extract the scale factor for the yukawa force compared to 10^8
         The function is performing the multi harmonics fit using the z axis
         :param y_focous: y position of beam from beam profiling
         :param x_focous: x position of beam from beam profiling
         :param large_bead: set to true if 7.6 um German beads are used (4.8um is used otherwise)
         :param lambda_par: lambda parameter for the Yukawa term
         :param freqs_array: search frequencies
         :param height: attractor height
         :param decimate: decimate data before the fit
         :param bandwidth: bandpass bandwidth
         :param noise_array: noise array for the fit
         :param bdf: bdf dataset to be used
         :return: amplitude, error
         """
        stroke = np.std(bdf.cant_pos[1] * 50) * np.sqrt(2) * 2  # stroke in y in micrometers
        cant_pos_x = np.mean(bdf.cant_pos[0])  # cantilever position in x for distance to sphere - in micrometers
        cant_pos_y = np.mean(bdf.cant_pos[1])  # cantilever position in x for distance to sphere - in micrometers
        if large_bead:
            separation = x_focous - aux.voltage_to_position(cant_pos_x) - 7.6 / 2 + 1
        else:
            separation = x_focous - aux.voltage_to_position(cant_pos_x) - 4.8 / 2
        time_sec = len(bdf.x2) / self.fsamp
        offset_y = (y_focous - 25 * 9.5) - (aux.voltage_to_position(cant_pos_y))
        # stroke = 100  # in microns
        # separation = 6.5  # in microns

        if not suppress_print:
            print('Large Bead: ', large_bead)
            print('Separation (face to face): ', separation)
            print('y-offset: ', offset_y)
            print('Stroke: ', stroke)
            print('Time: ', time_sec)

        if large_bead:
            template1 = force_vs_time(separation=separation * 1e-6, height=height, stroke=stroke * 1e-6,
                                      frequency=3,
                                      direction='z', lambda_par=lambda_par, offset_y=offset_y,
                                      yuk_or_grav="yuk", alpha=1e8, bead_size=3.8e-6)
        else:
            template1 = force_vs_time(separation=separation * 1e-6, height=height, stroke=stroke * 1e-6,
                                      frequency=3,
                                      direction='z', lambda_par=lambda_par, offset_y=offset_y,
                                      yuk_or_grav="yuk", alpha=1e8)

        template1 = np.array(template1[1]) * 1.85 / 1.55

        # data and scale preparation
        xx1 = bdf.z2
        tmp_scales = self.scale_Z2 * np.interp(freqs_array, self.tf_freq, self.tf_ffts[2])

        # find the mle
        m1_tmp = self.lc_i.find_mle_multiHarmoincs(x=xx1, template=template1, scales=tmp_scales,
                                                   signal_freqs=freqs_array, bandwidth=bandwidth,
                                                   noises=noise_array / noise_array[0],
                                                   decimate=decimate, **fit_kwargs)

        print('***************************************************')
        print('alpha mle: ', '{:.2e}'.format(m1_tmp.values[0]))
        print('sigma mle: ', '{:.2e}'.format(m1_tmp.values[2]))
        print('reduced chi2: ', m1_tmp.fval / (len(bdf.x2) - 1))

        return m1_tmp.values[0], m1_tmp.values[2], m1_tmp
