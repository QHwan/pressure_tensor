import cython
cimport cython
import numpy as np
cimport numpy as np
import sys
import math
from libc.math cimport sqrt, atan, log
from libc.stdlib cimport malloc, free
import time
from mdtraj.formats import XTCTrajectoryFile
from mdtraj.formats import TRRTrajectoryFile

from func cimport integral_range, cal_force, cal_pn, cal_pt, vec_dot, vec_minus


@cython.cdivision(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def main():

	# This line we declare the variables
	cdef int i, j, k, l, m
	cdef int frame, nFrame, nFrame_used, nMol, apm
	cdef int lenCoordVec
	cdef float kBT, T
	cdef float doh, dom
	cdef float t
	cdef float comx, comy, comz
	cdef float L, L2
	cdef float rMin, rMax, dr
	cdef float x, y, z, x0, y0, z0
	cdef float ri, rj, rij, ri2, rj2, rij2, ririj
	cdef float Din, Dout, sDin, sDout, Rin, Rout, Rin2, Rout2
	cdef float linn, linp, loutn, loutp, ll, l0
	cdef float la, lb, lap, lbp
	cdef float pN, pT, pK
	cdef float v
	cdef float pref
	cdef float cut

	cdef np.ndarray[float, ndim=3] coordtMat
	cdef np.ndarray[float, ndim=2] coordMat
	cdef np.ndarray[float, ndim=1] tVec
	cdef np.ndarray[float, ndim=1] rVec
	cdef np.ndarray[float, ndim=3] boxMat
	cdef np.ndarray[float, ndim=1] pkVec
	cdef np.ndarray[float, ndim=1] pnVec
	cdef np.ndarray[float, ndim=1] ptVec
	cdef np.ndarray[float, ndim=1] vVec

	cdef np.ndarray[float, ndim=3] fMat

	cdef np.ndarray[float, ndim=1] fLJVec
	cdef np.ndarray[float, ndim=1] fCVec
	fLJVec = np.zeros(3, dtype=np.float32)
	fCVec = np.zeros(3, dtype=np.float32)

#	cdef np.ndarray[float, ndim=2] mMat
#	cdef np.ndarray[float, ndim=1] mVec

	cdef float **iMat
	cdef float **jMat
	cdef float *iVec
	cdef float *jVec
	cdef float *oiVec
	cdef float *oijVec
	cdef float *fVec

	cdef int count
	count = 0

	

	# Program starts here
	# Setting the File variable
	trrFileName = sys.argv[1]
	mFileName = sys.argv[2]
	denFileName = sys.argv[3]
	oFileName = sys.argv[4]

	mMat = np.transpose(np.loadtxt(mFileName))
	mVec = mMat[1]

	dMat = np.transpose(np.loadtxt(denFileName, dtype=np.float32))
	rVec = dMat[0]
	dVec = dMat[1]

	with TRRTrajectoryFile(trrFileName) as trrFile:
		coordtMat, tVec, step, boxMat, lambdaVec = trrFile.read()
	trrFile.close()

	nFrame, nMol, xyz = np.shape(coordtMat)
	apm = 3
	nMol /= apm

	L = boxMat[0][0][0]
	L2 = L*0.5

	pref = 1.6602	# kJ/mol/nm3 -> MPa
	T = 300.
	kBT = 2.479*T/298

	cut = 4.0

	# Setting PMat
	dr = rVec[1]-rVec[0]
	rMin = dr*0.5
	rMax = rVec[len(rVec)-1]

	pkVec = np.zeros( (len(rVec)), dtype=np.float32)
	pnVec = np.zeros( (len(rVec)), dtype=np.float32)
	ptVec = np.zeros( (len(rVec)), dtype=np.float32)



	vVec = np.zeros(len(rVec), dtype=np.float32)
	for i in range (len(rVec)):
		vVec[i] = 4*math.pi/3.*(pow(rVec[i] + dr*0.5,3) - pow(rVec[i] - dr*0.5,3))


	fMat = np.zeros( (nMol,nMol,3), dtype=np.float32)
	fLJMat = np.zeros( (nMol, nMol, 3), dtype=np.float32)
	fCMat = np.zeros( (nMol, nMol, 3), dtype=np.float32)

	if len(rVec) != len(dVec):
		print 'rVec and denVec have different length'


	# Setting iVec and jVec
	iMat = <float **> malloc(apm * sizeof(float*))
	jMat = <float **> malloc(apm * sizeof(float*))
	for i in range (apm):
		iMat[i] = <float *> malloc(3 * sizeof(float))
		jMat[i] = <float *> malloc(3 * sizeof(float))
	for i in range (apm):
		for j in range (3):
			iMat[i][j] = 0
			jMat[i][j] = 0

	
	iVec = <float *> malloc(3 * sizeof(float))
	jVec = <float *> malloc(3 * sizeof(float))
	for i in range (3):
		iVec[i] = 0
		jVec[i] = 0

	oiVec = <float *> malloc(3 * sizeof(float))
	oijVec = <float *> malloc(3 * sizeof(float))
	for i in range (3):
		oiVec[i] = 0
		oijVec[i] =0 

	fVec = <float *> malloc(3 * sizeof(float))
	for i in range (3):
		fVec[i] = 0


	
	# Starting read coordinates
	frame = 0
	nFrame_used = 0
	for coordMat in coordtMat:

		if frame%100 == 0:
			print frame

		if frame == 100:
			break

		if mVec[frame] != nMol and nFrame != 1:
			frame += 1
			pass

		else:
			frame += 1
			nFrame_used += 1


			# Consider periodic boundary
			x0 = coordMat[0,0]; y0 = coordMat[0,1]; z0 = coordMat[0,2]
			for i in range (1, nMol):
				x = coordMat[apm*i,0]
				y = coordMat[apm*i,1]
				z = coordMat[apm*i,2]
				if x-x0 > L2:
					for j in range (apm):
						coordMat[apm*i+j,0] -= L
				elif x-x0 < -L2:
					for j in range (apm):
						coordMat[apm*i+j,0] += L
				if y-y0 > L2:
					for j in range (apm):
						coordMat[apm*i+j,1] -= L
				elif y-y0 < -L2:
					for j in range (apm):
						coordMat[apm*i+j,1] += L
				if z-z0 > L2:
					for j in range (apm):
						coordMat[apm*i+j,2] -= L
				elif z-z0 < -L2:
					for j in range (apm):
						coordMat[apm*i+j,2] += L

			# calculate com matrix using OW position
			comx = 0; comy = 0; comz = 0;
			for i in range (nMol):
				comx += coordMat[apm*i,0]
				comy += coordMat[apm*i,1]
				comz += coordMat[apm*i,2]
			comx /= nMol
			comy /= nMol
			comz /= nMol
			for i in range (apm*nMol):
				coordMat[i,0] -= comx
				coordMat[i,1] -= comy
				coordMat[i,2] -= comz

			# Calculate intermolecular force
			for i in range (nMol):
				for j in range (i+1,nMol):
					for k in range (apm):
						for l in range (3):
							iMat[k][l] = coordMat[apm*i+k,l]
							jMat[k][l] = coordMat[apm*j+k,l]

					fVec[0] = 0
					fVec[1] = 0
					fVec[2] = 0 

					cal_force(fVec, iMat, jMat, cut)

					fMat[i,j,0] = fVec[0]
					fMat[i,j,1] = fVec[1]
					fMat[i,j,2] = fVec[2]




			# Calculate Pressure
			for i in range (nMol):

				for j in range (3):
					oiVec[j] = coordMat[apm*i,j]

				ri2 = 0.
				for j in range (3):
					ri2 += oiVec[j]*oiVec[j]



				for j in range (i+1, nMol):

					for l in range (3):
						oijVec[l] = coordMat[apm*j,l] - oiVec[l]

					rij2=0.
					ririj=0.
					for l in range (3):
						rij2 += oijVec[l]*oijVec[l]
						ririj += oiVec[l]*oijVec[l]

					rij = sqrt(rij2)

					for k in range (len(rVec)):
						r = rVec[k]
						Rin = r-dr*0.5
						Rout = r+dr*0.5
						Rin2 = Rin*Rin
						Rout2 = Rout*Rout


						# Check Din and Dout
						Din = ririj**2 - rij2*(ri2 - Rin2)
						Dout = ririj**2 - rij2*(ri2 - Rout2)

						if Dout <= 0:
							pass
						else:
							ll = -ririj/rij2
							loutp = ll + sqrt(Dout)/rij2
							loutn = ll - sqrt(Dout)/rij2

							if loutn>1 or loutp<0:
								pass
							else:
								
								if Din > 0:
									linp = ll + sqrt(Din)/rij2
									linn = ll - sqrt(Din)/rij2
								else:
									linp = -10
									linn = -10


								if linn<0 and linp>1:
									pass
								else:
									# choose integral range
									la, lb, lap, lbp = integral_range(Din, loutn, linn, linp, loutp)

									fVec[0] = fMat[i,j,0]
									fVec[1] = fMat[i,j,1]
									fVec[2] = fMat[i,j,2]

									f = (fVec[0]*oijVec[0] + fVec[1]*oijVec[1] + fVec[2]*oijVec[2])/rij
									l0 = sqrt(ri2 - (ririj/rij)**2)

									pnVec[k] += (rij*(lb-la))
									pnVec[k] += -l0 * atan(sqrt(rij2*lb*lb + 2*ririj*lb + ri2 - l0*l0)/l0)
									pnVec[k] -= -l0 * atan(sqrt(rij2*la*la + 2*ririj*la + ri2 - l0*l0)/l0)
									pnVec[k] *= f
									if l0*l0 > rij2*lb*lb + 2*ririj*lb + ri2:
										print l0*l0, rij2*lb*lb + 2*ririj*lb + ri2
										exit(1)

									#pnVec[k] += cal_pn(fVec, oiVec, oijVec, la, lb)
									#ptVec[k] += cal_pt(fVec, oiVec, oijVec, la, lb)
									if lap >= 0 and lbp >= 0 and lbp > lap:
										pnVec[k] += (rij*(lbp-lap))
										pnVec[k] += -l0 * atan(sqrt(rij2*lbp*lbp + 2*ririj*lbp + ri2 - l0*l0)/l0)
										pnVec[k] -= -l0 * atan(sqrt(rij2*lap*lap + 2*ririj*lap + ri2 - l0*l0)/l0)
										pnVec[k] *= f

										#pnVec[k] -= cal_pn(fVec, oiVec, oijVec, lap, lbp)
									#	ptVec[k] += cal_pt(fVec, oiVec, oijVec, lap, lbp)

									'''
									if np.isnan(ptVec[k]):
										a = oiVec[0]*fVec[1] - oiVec[1]*fVec[0]
										b = oijVec[0]*fVec[1] - oijVec[1]*fVec[0]
										c = oiVec[0]*oijVec[1] - oiVec[1]*oijVec[0]
										d = oiVec[0]*oiVec[0] + oiVec[1]*oiVec[1]
										e = 2*(oiVec[0]*oijVec[0] + oiVec[1]*oijVec[1])
										f = oijVec[0]*oijVec[0] + oijVec[1]*oijVec[1]
										bb = sqrt(-4*d*f+e*e)


										print d+e*la+f*la*la
										print d+e*lb+f*lb*lb
										print b/2/f*( log(d+e*lb+f*lb*lb) - log(d+e*la+f*la*la) ) 
										print 'bb = {}'.format(bb)
										print e+2*f*lb
										print atanh((e+2*f*lb)/bb)
										print -(2*a*f-b*e)/f/bb * (atanh((e+2*f*lb)/bb) - atanh((e+2*f*la)/bb))

										print 'r = {}'.format(rVec[k])
										print oiVec[0], oiVec[1], oiVec[2], sqrt(ri2)
										print oiVec[0]+oijVec[0], oiVec[1]+oijVec[1], oiVec[2]+oijVec[2]
										print Dout, Din
										print loutn, linn, linp, loutp
										print la, lb, lap, lbp
										exit(1)
									'''




	# prepare pK
	for i in range (len(rVec)):
		pkVec[i] = pref*kBT*dVec[i]
	
	for i in range (len(rVec)):
		pnVec[i] *= pref/nFrame_used/vVec[i]
		ptVec[i] *= pref/nFrame_used/vVec[i]
	
		
	print 'nFrame_used = {}'.format(nFrame_used)
	
	# coordt loop is over
	oMat = []
	oMat.append(rVec)
	oMat.append(pkVec)
	oMat.append(ptVec)
	oMat.append(pnVec)
	oMat.append(pkVec+ptVec+pnVec)
	oMat = np.transpose(oMat)

	
	# Save file
	np.savetxt(oFileName,oMat,fmt='%7f')


	# Free everything
	for i in range (apm):
		free(iMat[i])
		free(jMat[i])
	free(iMat)
	free(jMat)

	free(iVec)
	free(jVec)

	free(oiVec)
	free(oijVec)

	free(fVec)

	print count

		


							






