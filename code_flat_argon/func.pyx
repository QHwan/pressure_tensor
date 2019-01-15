import numpy as np
cimport numpy as np
import math
from libc.math cimport sqrt, log, atan
from libc.stdlib cimport malloc, free

cdef vec_minus(float *ijVec, float *iVec, float *jVec):
	cdef int i

	for i in range (3):
		ijVec[i] = jVec[i] - iVec[i]


cdef vec_dot(float *iVec, float *jVec):
	cdef int i
	cdef float d

	d = 0
	for i in range (3):
		d += iVec[i] * jVec[i]
	
	return d


cdef integral_range(float Din, float loutn, float linn, float linp, float loutp):
	cdef float la, lb, lap, lbp

	if linn<0 and loutn<0:
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
	elif Din<0:
		la = loutn
		lb = loutp
		lap = -10
		lbp = -10
		if loutn<0:
			la = 0
		if loutp>1:
			lb = 1
	else:
		la = loutn
		lb = linn
		lap = linp
		lbp = loutp
		if loutn<0:
			la = 0
		if loutp>1:
			lb = 1
	return la, lb, lap, lbp


cdef cal_LJ_force(float *ijVec, float rij, float sig6, float eps):
	cdef float f
	cdef float fx, fy, fz
	cdef np.ndarray[float, ndim=1] fLJVec

	if sig6==0 or eps==0:
		fLJVec = np.zeros(3, dtype=np.float32)
	else:

		f = 12*sig6*sig6/pow(rij,13) - 6*sig6/pow(rij,7)
		f *= 4*eps

		fLJVec = np.zeros(3, dtype=np.float32)
		fLJVec[0] = f*ijVec[0]/rij
		fLJVec[1] = f*ijVec[1]/rij
		fLJVec[2] = f*ijVec[2]/rij

	return fLJVec


cdef cal_C_force(float *ijVec, float rij, float qi, float qj):
	cdef float f
	cdef float pref
	cdef float fx, fy, fz
	cdef np.ndarray[float, ndim=1] fCVec

	if qi==0 or qj==0:
		fCVec = np.zeros(3, dtype=np.float32)
	else:
		pref = 138.935458

		f = pref*qi*qj/rij/rij/rij

		fCVec = np.zeros(3, dtype=np.float32)
		fCVec[0] = f*ijVec[0]
		fCVec[1] = f*ijVec[1]
		fCVec[2] = f*ijVec[2]

	return fCVec


cdef cal_force(float *fVec, float *ijVec, float rij):
	cdef int apm
	cdef float sig, eps, sig6
	cdef float f, pref

	sig = 0.3405
	sig6 = sig**6
	eps = 0.99607

	apm = 1

	f = 12*sig6*sig6/pow(rij,13) - 6*sig6/pow(rij,7)
	f *= 4*eps

	fVec[0] = f*ijVec[0]/rij
	fVec[1] = f*ijVec[1]/rij
	fVec[2] = f*ijVec[2]/rij

			






cdef pn_core(a, b, c, d, e, f, l):
	pn = 2*c*f*l
	pn += (b*f - c*e)*log(d + e*l + f*l*l)
	pn += 2*atan( (e+2*f*l)/sqrt(4*d*f-e*e) )/sqrt(4*d*f - e*e) * (f*(2*a*f-b*e) + c*(e*e-2*d*f))
	pn *= 0.5/(f*f)
	return pn

cdef cal_pn(float *fVec, float *iVec, float *ijVec, la, lb):
	cdef int i
	cdef float a, b, c, d, e, f
	cdef float rifij, rijfij, ririj, riri, rijrij
	cdef float pn, pnb, pna

	cdef float aa, bb, bc

	rifij = 0.
	rijfij = 0.
	ririj = 0.
	riri = 0.
	rijrij = 0.
	for i in range (3):
		rifij += iVec[i]*fVec[i]
		rijfij += ijVec[i]*fVec[i]
		ririj += iVec[i]*ijVec[i]
		rijrij += ijVec[i]*ijVec[i]
		riri += iVec[i]*iVec[i]

	a = rifij * ririj
	b = rijfij * ririj + rifij * rijrij
	c = rijfij * rijrij
	d = riri
	e = 2*ririj
	f = rijrij

	aa = b*f - c*e

	bc = 4*d*f - e*e

	pnb = 2*c*f*lb
	pnb += (b*f - c*e)*log(d + e*lb + f*lb*lb)
	pnb += 2*atan( (e+2*f*lb)/sqrt(4*d*f-e*e) )/sqrt(4*d*f - e*e) * (f*(2*a*f-b*e) + c*(e*e-2*d*f))
	pnb *= 0.5/(f*f)

	pna = 2*c*f*la
	pna += (b*f - c*e)*log(d + e*la + f*la*la)
	pna += 2*atan( (e+2*f*la)/sqrt(4*d*f-e*e) )/sqrt(4*d*f - e*e) * (f*(2*a*f-b*e) + c*(e*e-2*d*f))
	pna *= 0.5/(f*f)

	pn = pnb - pna

	return pn


cdef cal_pt(float *fVec, float *iVec, float *ijVec, la, lb):
	cdef float a, b, c, d, e, f
	cdef float pt, ptb, pta

	cdef float bb

	a = iVec[0]*fVec[1] - iVec[1]*fVec[0]
	b = ijVec[0]*fVec[1] - ijVec[1]*fVec[0]
	c = iVec[0]*ijVec[1] - iVec[1]*ijVec[0]
	d = iVec[0]*iVec[0] + iVec[1]*iVec[1]
	e = 2*(iVec[0]*ijVec[0] + iVec[1]*ijVec[1])
	f = ijVec[0]*ijVec[0] + ijVec[1]*ijVec[1]


	if 4*d*f-e*e <= 0 :
		pt = 0.
	else:
		bb = sqrt(4*d*f-e*e)
	
		pt = b/2/f*(log(d+e*lb+f*lb*lb) - log(d+e*la+f*la*la))
		pt += (2*a*f-b*e)/f/bb * (atan((e+2*f*lb)/bb) - atan((e+2*f*la)/bb))
		pt *= c

	return pt

