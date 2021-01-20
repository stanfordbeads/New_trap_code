import numpy as np
from iminuit import Minuit


def shift_correct(input_pos, idx):
    output_pos = np.insert(input_pos, 0, np.linspace(0, 0, idx))
    return output_pos


def gaussian(data_x, params=list):
    norm = (1 / ((1 / 2 * params[2]) * np.sqrt(np.pi * 2)))
    return params[0] * norm * np.exp(-(np.subtract(data_x, params[1]) ** 2 / (2 * params[2] ** 2))) + params[3]


def gaussian_cdf(data_x, params=list):
    # for normalization a 1/sigma could be needed
    return params[0] * 0.5 * (1 + scipy.special.erf((data_x - params[1]) / (np.sqrt(2) * params[2]))) + params[3]


def linear(data_x, params=list):
    return params[0] * data_x + params[1]


def chisquare_1d(function, functionparams, data_x, data_y, data_y_error):
    chisquarevalue = np.sum(np.power(np.divide(np.subtract(function(data_x, functionparams), data_y), data_y_error), 2))
    ndf = len(data_y) - len(functionparams)
    # print(ndf)
    return chisquarevalue, ndf


def chisquare_linear(a, b):
    return chisquare_1d(function=linear, functionparams=[a, b], data_x=data_x, data_y=data_y,
                        data_y_error=data_y_error)[0]


# calibration of the voltage - position conversion

def voltage_to_position(voltage, slope=0.019834000085488412, offset=-0.0015000315197539749, redo=False):
    if redo:
        pos_list = np.asarray([-0.007, 4.968, 9.91])
        y_err = np.asarray([0.01, 0.01, 0.01])
        val = np.asarray([0, 250, 500])
        m2 = Minuit(chisquare_linear,
                    a=100,
                    b=0,
                    errordef=1,
                    print_level=1,
                    data_x=val, fix_data_x=True,
                    data_y=pos_list, fix_data_y=True,
                    data_y_error=y_err, fix_data_y_error=True)
        m2.migrad()
        slope = m2.values["a"]
        offset = m2.values["b"]
    position = (voltage - offset) / slope
    return position
