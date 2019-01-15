import numpy as np
cimport numpy as np

cdef vec_minus(float *ijVec, float *iVec, float *jVec)
cdef vec_dot(float *iVec, float *jVec)

cdef integral_range(float Din, float loutn, float linn, float linp, float loutp)
cdef cal_force(float *fVec, float **iMat, float **jMat)

cdef cal_pn(float *fVec, float *iVec, float *jVec, la, lb)
cdef cal_pt(float *fVec, float *iVec, float *jVec, la, lb)


