import time
import numpy
import itertools
from scipy.misc import comb
from astropy.io import fits
from aotools.functions import circle
from matplotlib import pyplot; pyplot.ion()
from capt.misc_functions.cross_cov import cross_cov
from capt.roi_functions.gamma_vector import gamma_vector
from capt.misc_functions.make_pupil_mask import make_pupil_mask
from capt.roi_functions.roi_referenceArrays import roi_referenceArrays
from capt.misc_functions.mapping_matrix import get_mappingMatrix, covMap_superFast, arrayRef
from joblib import Parallel, delayed, parallel_backend
#from dask.distributed import Client


def inner_loop(j, i, roi_ones_arange, mm_subapPos,sa_mm,sb_mm, mm, subap1_comb_shift, subap2_comb_shift, roi_axis, mapping_type, shwfs_centroids, wfs1_n_subap, wfs2_n_subap, roi_cov_xx, roi_cov_yy):
        #print('j={}'.format(j))
        roi_loc = numpy.where(roi_ones_arange==j)
        roi_baseline = mm_subapPos[i, roi_loc[0], roi_loc[1]]
        
        subaps1 = sa_mm[:, roi_baseline][numpy.where(mm[:, roi_baseline]==1)] + subap1_comb_shift
        subaps2 = sb_mm[:, roi_baseline][numpy.where(mm[:, roi_baseline]==1)] + subap2_comb_shift
        num_subaps = subaps1.shape[0]
        
        # stop
        if roi_axis!='y':
                if mapping_type=='mean':
                        cova = numpy.mean((shwfs_centroids[subaps1] * (shwfs_centroids[subaps2])).sum(1)/(shwfs_centroids.shape[1]-1))
                if mapping_type=='median':
                        cova = numpy.median((shwfs_centroids[subaps1] * (shwfs_centroids[subaps2])).sum(1)/(shwfs_centroids.shape[1]-1))
                roi_cov_xx[roi_loc[0], roi_loc[1]] = cova

                
        if roi_axis!='x':
                if mapping_type=='mean':
                        cova = numpy.mean((shwfs_centroids[subaps1+wfs1_n_subap] * (shwfs_centroids[subaps2+wfs2_n_subap])).sum(1)/(shwfs_centroids.shape[1]-1))
                if mapping_type=='median':
                        cova = numpy.median((shwfs_centroids[subaps1+wfs1_n_subap] * (shwfs_centroids[subaps2+wfs2_n_subap])).sum(1)/(shwfs_centroids.shape[1]-1))
                        
                roi_cov_yy[roi_loc[0], roi_loc[1]] = cova

        #return (roi_loc[0], roi_loc[1], covax, covay)


def calculate_roi_covariance(shwfs_centroids, gs_pos, pupil_mask, tel_diam, roi_belowGround, roi_envelope, roi_axis, mapping_type):
	"""Takes SHWFS centroids and directly calculates the covariance map ROI (does not require going via covariance matrix).

	Parameters:
		shwfs_centroids (ndarray): SHWFS centroid measurements.
		gs_pos (ndarray): GS asterism in telescope FoV.
		pupil_mask (ndarray): mask of SHWFS sub-apertures within the telescope pupil.
		tel_diam (float): diameter of telescope pupil.
		roi_belowGround (int): number of sub-aperture separations the ROI encapsulates 'below-ground'.
		roi_envelope (int): number of sub-aperture separations either side of the ROI.
		roi_axis (str): in which axis to express ROI ('x', 'y', 'x+y' or 'x and y')
		mapping_type (str): how to calculate overall sub-aperture separation covariance ('mean' or 'median')

	Returns:
		roi_covariance (ndarray): covariance map ROI.
		time_taken (float): time taken to complete calculation."""

        print('roi_axis = {}'.format(roi_axis))
	covMapDim = pupil_mask.shape[0] * 2 -1
	n_subap = numpy.array([int(pupil_mask.sum())]*gs_pos.shape[0])
	mm, sa_mm, sb_mm, allMapPos, selector, xy_separations = roi_referenceArrays(
				numpy.rot90(pupil_mask,2), gs_pos, tel_diam, roi_belowGround, roi_envelope)

	timeStart = time.time()

	#subtracts mean at each sub-aperture axis (first step in calculating cross-covariance).
	shwfs_centroids = (shwfs_centroids - shwfs_centroids.mean(0)).T

	if roi_axis=='x' or roi_axis=='y' or roi_axis=='x+y':
		roi_covariance = numpy.zeros((allMapPos.shape[0]*allMapPos.shape[1], allMapPos.shape[2]))
	if roi_axis=='x and y':
		roi_covariance = numpy.zeros((allMapPos.shape[0]*allMapPos.shape[1], allMapPos.shape[2]*2))

	wfs1_n_subap = n_subap[0]
	wfs2_n_subap = n_subap[0]

	mm_subapPos = allMapPos[:, :, :, 1] + allMapPos[:, :, :, 0] * covMapDim
        #debug
        print('allMapPos.shape', allMapPos.shape)
        mapPos0 = allMapPos.shape[0]
        mapPos1 = allMapPos.shape[1]
        mapPos2 = allMapPos.shape[2]
        mapPos3 = allMapPos.shape[3]
	for i in range(mapPos0):
                tim1 = time.time()
		roi_ones = numpy.ones(allMapPos[i,:,:,0].shape)
  		roi_ones[numpy.where(allMapPos[i,:,:,0]==2*covMapDim)] = 0
  		num_roi_baselines = int(roi_ones.sum())
  		arange_baselines = numpy.arange(num_roi_baselines) + 1
  		roi_ones_arange = roi_ones.copy()
  		roi_ones_arange[roi_ones==1] = arange_baselines
  		av = numpy.ones(roi_ones.shape)
  
		#integer shift for each GS combination 
		subap1_comb_shift = selector[i][0]*2*wfs1_n_subap
                subap2_comb_shift = selector[i][1]*2*wfs1_n_subap

		if roi_axis!='y':
			roi_cov_xx = numpy.zeros(roi_ones.shape)

		if roi_axis!='x':
			roi_cov_yy = numpy.zeros(roi_ones.shape)
                #debug
                print('num_roi_baselines for map column {} is {}'.format(i, num_roi_baselines))
                
                #parallel_loop for j in range(1, num_roi_baselines+1)
                print('starting 1st parallel inner loop')
                t1 = time.time()
                roi_cov_nores = Parallel(prefer='threads', n_jobs=32)(delayed(inner_loop)(j, i, roi_ones_arange, mm_subapPos,sa_mm,sb_mm, mm, subap1_comb_shift, subap2_comb_shift, roi_axis, mapping_type, shwfs_centroids, wfs1_n_subap, wfs2_n_subap, roi_cov_xx, roi_cov_yy) for j in range(1, 33))
                t2 = time.time()
                print('parallel inner loop finished in {} seconds'.format(t2-t1))
                #print('doing remaining loop in serial')
                #roi_cov_nores = Parallel(prefer='threads', n_jobs=16)(delayed(inner_loop)(j, i, roi_ones_arange, mm_subapPos,sa_mm,sb_mm, mm, subap1_comb_shift, subap2_comb_shift, roi_axis, mapping_type, shwfs_centroids, wfs1_n_subap, wfs2_n_subap, roi_cov_xx, roi_cov_yy) for j in range(17, 33))
                t1 = time.time()
                roi_cov_nores = Parallel(prefer='threads', n_jobs=16)(delayed(inner_loop)(j, i, roi_ones_arange, mm_subapPos,sa_mm,sb_mm, mm, subap1_comb_shift, subap2_comb_shift, roi_axis, mapping_type, shwfs_centroids, wfs1_n_subap, wfs2_n_subap, roi_cov_xx, roi_cov_yy) for j in range(33,49))
                t2 = time.time()
                print('2nd parallel inner loop finished in {} seconds'.format(t2-t1))

		if roi_axis=='x':
			roi_covariance[i*mapPos1:(i+1)*mapPos1] = roi_cov_xx
		if roi_axis=='y':
			roi_covariance[i*mapPos1:(i+1)*mapPos1] = roi_cov_yy
		if roi_axis=='x+y':
			roi_covariance[i*mapPos1:(i+1)*mapPos1] = (roi_cov_xx+roi_cov_yy)/2.
		if roi_axis=='x and y':
			roi_covariance[i*mapPos1:(i+1)*mapPos1] = numpy.hstack((roi_cov_xx, roi_cov_yy))
                tim4 = time.time()
                print('outer loop iteration took:', tim4-tim1, 's')
                #numpy.savetxt('covtimes_{}.txt'.format(i),covtimes,fmt=['%f','%f'] )
                #numpy.savetxt('subapshapes_{}.txt'.format(i),subapshapes, fmt=['%d','%d'])

	timeStop = time.time()
	time_taken = timeStop - timeStart

	return roi_covariance, time_taken






if __name__=='__main__':
        import sys
        from capt.misc_functions.dasp_cents_reshape import dasp_cents_reshape
        idir = 'dasp-centroids/'
        if len(sys.argv) > 1:
                idir = sys.argv[1] 
        print('using input directory for slopes: {}, to change quit and run again with {} <dir>'.format(idir,sys.argv[0] ))

        print('preparing parameters')
	n_wfs = 6
        r = 60
	gs_pos = numpy.array([(r, 0), (r*numpy.cos(2*numpy.pi/6),r*numpy.sin(2*numpy.pi/6) ), (r*numpy.cos(2*2*numpy.pi/6),r*numpy.sin(2*2*numpy.pi/6) ),(r*numpy.cos(3*2*numpy.pi/6),r*numpy.sin(3*2*numpy.pi/6) ), (r*numpy.cos(4*2*numpy.pi/6),r*numpy.sin(4*2*numpy.pi/6) ), (r*numpy.cos(5*2*numpy.pi/6),r*numpy.sin(5*2*numpy.pi/6) ) ])
	# gs_pos = numpy.array(([0,-40], [0, 0]))
	tel_diam = 39.3
        obs_diam = 11.1
	roi_belowGround = 0
	roi_envelope = 0
	nx_subap = numpy.array([80]*n_wfs)
	n_subap = numpy.array([4632]*n_wfs)

        print('creating pupil mask')
	pupil_mask = make_pupil_mask('circle', n_subap, nx_subap[0], 
			obs_diam, tel_diam)

        print('creating ROI reference arrays')
	onesMat, wfsMat_1, wfsMat_2, allMapPos, selector, xy_separations = roi_referenceArrays(
                pupil_mask, gs_pos, tel_diam, roi_belowGround, roi_envelope)

        print('Reading centroids')	
        centslist = [fits.getdata(idir+'saveOutput_{}b0.fits'.format(i+1)).byteswap() for i in range(n_wfs)]
        shwfs_centroids = numpy.stack(centslist) 
        shwfs_centroids = dasp_cents_reshape(shwfs_centroids, pupil_mask, n_wfs)
#	shwfs_centroids = shwfs_centroids[:, reduce_cents==2]

	roi_axis = 'x and y'
	mapping_type = 'mean'

        print('calling calculate_roi_covariance')
	nr, nt = calculate_roi_covariance(shwfs_centroids, gs_pos, pupil_mask, tel_diam, roi_belowGround, roi_envelope, roi_axis, mapping_type)
	print('Time taken: {}'.format(nt))

#	pyplot.figure()
#	pyplot.imshow(nr)
