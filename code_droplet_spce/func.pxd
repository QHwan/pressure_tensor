import numpy as np
cimport numpy as np

cdef vec_minus(double *ijVec, double *iVec, double *jVec)
cdef vec_dot(double *iVec, double *jVec)

cdef integral_range(double Din, double loutn, double linn, double linp, double loutp)
cdef void cal_force(double *fVec, double **iMat, double **jMat, double cut, int apm)

cdef double cal_pn(double *fVec, double *iVec, double *jVec, double la, double lb, double ri2, double rij2, double ririj)
cdef double cal_pt(double *fVec, double *iVec, double *jVec, double la, double lb)


