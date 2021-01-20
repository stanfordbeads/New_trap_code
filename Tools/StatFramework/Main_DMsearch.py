import numpy as np
import matplotlib.pyplot as plt

from likelihood_calculator import likelihood_analyser
import BeadDataFile
from likelihood_calculator import dm_search

fnames = [r"C:\Users\nadav\PythonCode\SomeData\20191107\DMdata\Discharge_{}.h5".format(i) for i in range(100, 130)]
bdfs = [BeadDataFile.BeadDataFile(fname_) for fname_ in fnames]
DManalyzer = dm_search.DMAnalyser()
DManalyzer.BDFs = bdfs

# noise calculation
DManalyzer.estimate_noise()  # estimate noise of x2 and x3
# DManalyzer.plot_dataset(4)  # plot one dataset

# DManalyzer.get_delta_alpha(bdf_i=3, alpha_frequency=1)
# DManalyzer.get_delta_alpha(bdf_i=10, alpha_frequency=1)

alphas_freq = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 2, 3, 4, 5, 6, 7, 8]
alphas_freq = np.logspace(-3, 0.7, 100)
limit = [DManalyzer.get_sensitivity(alpha_frequency=freq) for freq in alphas_freq]

_, ax = plt.subplots()
ax.loglog(alphas_freq, limit, '.--')
ax.set(xlabel='freq[Hz]', ylabel='delta_alpha')
plt.show()
