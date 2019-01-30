import numpy as np
cimport numpy as np
import math
from libc.math cimport log, sqrt, atan
from libc.stdlib cimport malloc, free


cdef vec_minus(double *ijVec, double *iVec, double *jVec):
	cdef int i

	for i in range (3):
		ijVec[i] = jVec[i] - iVec[i]


cdef vec_dot(double *iVec, double *jVec):
	cdef int i
	cdef double d

	d = 0
	for i in range (3):
		d += iVec[i] * jVec[i]
	
	return d


cdef integral_range(double Din, double loutn, double linn, double linp, double loutp):
	cdef double la, lb, lap, lbp

	if Din <= 0:
		la = loutn
		lb = loutp
		lap = -10
		lbp = -10
		if loutn<0:
			la = 0
		if loutp>1:
			lb = 1
	else:
		if loutn<0 and linn<0:
			la = linp
			lb = loutp
			lap = -10
			lbp = -10
			if linp<0:
				la = 0
			if loutp>1:
				lb = 1
		elif linp>1 and loutp>1:
			la = loutn
			lb = linn
			lap = -10
			lbp = -10
			if loutn<0:
				la = 0
			if linn>1:
				lb = 1
		else:
			la = loutn
			lb = linn
			lap = linp
			lbp = loutp
			if loutn<0:
				la = 0
			if loutp>1:
				lbp = 1

	return la, lb, lap, lbp


cdef void cal_force(double *fVec, double **iMat, double **jMat, double cut, int apm):
	cdef double sig, eps, sig6
	cdef double qo, qh
	cdef double f, pref

	cdef np.ndarray[double, ndim=1, mode="c"] qVec

	sig = 0.316557
	sig6 = sig*sig*sig*sig*sig*sig
	eps = 0.65019

	pref = 138.935458
	qo = -0.8476
	qh = 0.4238

	qVec = np.array([qo, qh, qh])

	# considering charge group
	rij = sqrt((jMat[0][0]-iMat[0][0])*(jMat[0][0]-iMat[0][0])
		  + (jMat[0][1]-iMat[0][1])*(jMat[0][1]-iMat[0][1]) 
		  + (jMat[0][2]-iMat[0][2])*(jMat[0][2]-iMat[0][2]))

	if rij > cut:
		fVec[0] = 0
	else:
		f = 24*eps*sig6/(rij*rij*rij*rij*rij*rij*rij*rij) *  (2*sig6/(rij*rij*rij*rij*rij*rij) - 1)

		fVec[0] += f*(jMat[0][0]-iMat[0][0])
		fVec[1] += f*(jMat[0][1]-iMat[0][1])
		fVec[2] += f*(jMat[0][2]-iMat[0][2])

		for i in range (apm):
			for j in range (apm):
				rij = sqrt((jMat[j][0]-iMat[i][0])*(jMat[j][0]-iMat[i][0]) 
					  + (jMat[j][1]-iMat[i][1])*(jMat[j][1]-iMat[i][1]) 
					  + (jMat[j][2]-iMat[i][2])*(jMat[j][2]-iMat[i][2]))

				f = pref*qVec[i]*qVec[j]/(rij*rij*rij)

				fVec[0] += f*(jMat[j][0]-iMat[i][0])
				fVec[1] += f*(jMat[j][1]-iMat[i][1])
				fVec[2] += f*(jMat[j][2]-iMat[i][2])


cdef double cal_pn(double *fVec, double *iVec, double *ijVec, double la, double lb, double ri2, double rij2, double ririj):
	cdef double a, b, c, d, e, f
	cdef double rifij, rijfij
	cdef double pn = 0
	cdef double bb

	rifij = 0.
	rijfij = 0.
	rifij = iVec[0]*fVec[0] + iVec[1]*fVec[1] + iVec[2]*fVec[2]
	rijfij = ijVec[0]*fVec[0] + ijVec[1]*fVec[1] + ijVec[2]*fVec[2]

	a = rifij * ririj
	b = rijfij * ririj + rifij * rij2
	c = rijfij * rij2
	d = ri2
	e = 2*ririj
	f = rij2

	if 4*d*f - e*e <= 0:
		return pn
	else:
		bb = 1/sqrt(4*d*f-e*e)

		pn = 2*c*f*(lb-la) 
		pn += (b*f - c*e)*( log( (d+e*lb+f*lb*lb)/(d+e*la+f*la*la) ) )  
		pn += 2*bb*(f*(2*a*f-b*e)+c*(e*e-2*d*f))* (atan((e+2*f*lb)*bb) - atan((e+2*f*la)*bb))
		pn *= 0.5/(f*f)

	return pn



cdef double cal_pt(double *fVec, double *iVec, double *ijVec, double la, double lb):
	cdef double a, b, c, d, e, f
	cdef double pt = 0
	cdef double bb

	a = iVec[0]*fVec[1] - iVec[1]*fVec[0]
	b = ijVec[0]*fVec[1] - ijVec[1]*fVec[0]
	c = iVec[0]*ijVec[1] - iVec[1]*ijVec[0]
	d = iVec[0]*iVec[0] + iVec[1]*iVec[1]
	e = 2*(iVec[0]*ijVec[0] + iVec[1]*ijVec[1])
	f = ijVec[0]*ijVec[0] + ijVec[1]*ijVec[1]

	if 4*d*f-e*e <= 0 :
		return pt
	else:
		bb = 1/sqrt(4*d*f-e*e)

		pt = 0.5*b/f*( log((d+e*lb+f*lb*lb)/(d+e*la+f*la*la)) ) 
		pt += (2*a*f-b*e)/f*bb * (atan((e+2*f*lb)*bb) - atan((e+2*f*la)*bb))
		pt *= c

	return pt
