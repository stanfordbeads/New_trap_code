cimport numpy as np
cimport cython
from iminuit.util import describe, make_func_code
from libc.math cimport pow, M_PI, sin

cdef class LeastSquares:
    
    cdef np.ndarray data_x
    cdef np.ndarray data_y
    cdef int ndata
    
    def __init__(self, data_x, data_y):
        self.ndata = len(data_y)
        self.data_x= data_x
        self.data_y = data_y

    @cython.embedsignature(True)  # put function signature in pydoc so `describe` can extract it
    cpdef float compute(self,double A, double f, double phi):
        cdef float res = 0
        cdef int i
        cdef double val
        for i in range(self.ndata):
            x_val = self.data_x[i]
            val = A*sin(2*M_PI*f*x_val+phi)
            res += pow(self.data_y[i] - val, 2)
        return res
