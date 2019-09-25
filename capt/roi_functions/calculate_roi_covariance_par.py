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
from joblib import Parallel, delayed


def inner_loop(j,roi_ones_arange, mm_subapPos,sa_mm,sb_mm, mm, subap1_comb_shift, subap2_comb_shift, roi_axis, mapping_type, shwfs_centroids ):
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

        return roi_cov_xx, roi_cov_yy


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
	for i in range(allMapPos.shape[0]):
                tim1 = time.time()
		roi_ones = numpy.ones(allMapPos[i,:,:,0].shape)
                #tim2= time.time()
		roi_ones[numpy.where(allMapPos[i,:,:,0]==2*covMapDim)] = 0
                #tim3= time.time()

		num_roi_baselines = int(roi_ones.sum())
                #tim4= time.time()
		arange_baselines = numpy.arange(num_roi_baselines) + 1
                #tim5= time.time()
		roi_ones_arange = roi_ones.copy()
                #tim6= time.time()
		roi_ones_arange[roi_ones==1] = arange_baselines
                #tim7=time.time()
		av = numpy.ones(roi_ones.shape)
                #tim8= time.time()

		#integer shift for each GS combination 
		subap1_comb_shift = selector[i][0]*2*wfs1_n_subap
                #tim9= time.time()
		subap2_comb_shift = selector[i][1]*2*wfs1_n_subap
                #tim10=time.time()

		if roi_axis!='y':
			roi_cov_xx = numpy.zeros(roi_ones.shape)

		if roi_axis!='x':
			roi_cov_yy = numpy.zeros(roi_ones.shape)
                #debug
                print('num_roi_baselines for map column {} is {}'.format(i, num_roi_baselines))
                #subapshapes=numpy.zeros((num_roi_baselines,2))
                #covtimes=numpy.zeros((num_roi_baselines,2))
                
                #parallel_loop
		roi_cov_xx, roi_cov_yy = Parallel(n_jobs=-1, backend="threading")(delayed(inner_loop)(j, roi_ones_arange, mm_subapPos,sa_mm,sb_mm, mm, subap1_comb_shift, subap2_comb_shift, roi_axis, mapping_type, shwfs_centroids) for j in range(1, num_roi_baselines+1))
		if roi_axis=='x':
			roi_covariance[i*allMapPos.shape[1]:(i+1)*allMapPos.shape[1]] = roi_cov_xx
		if roi_axis=='y':
			roi_covariance[i*allMapPos.shape[1]:(i+1)*allMapPos.shape[1]] = roi_cov_yy
		if roi_axis=='x+y':
			roi_covariance[i*allMapPos.shape[1]:(i+1)*allMapPos.shape[1]] = (roi_cov_xx+roi_cov_yy)/2.
		if roi_axis=='x and y':
			roi_covariance[i*allMapPos.shape[1]:(i+1)*allMapPos.shape[1]] = numpy.hstack((roi_cov_xx, roi_cov_yy))
                tim4 = time.time()
                print('outer loop iteration took:', tim4-tim1, 's')
                #numpy.savetxt('covtimes_{}.txt'.format(i),covtimes,fmt=['%f','%f'] )
                #numpy.savetxt('subapshapes_{}.txt'.format(i),subapshapes, fmt=['%d','%d'])

	timeStop = time.time()
	time_taken = timeStop - timeStart

	return roi_covariance, time_taken






if __name__=='__main__':
	n_wfs = 3
	gs_pos = numpy.array(([0,-40], [0, 0], [30,0]))
	# gs_pos = numpy.array(([0,-40], [0, 0]))
	tel_diam = 4.2
	roi_belowGround = 6
	roi_envelope = 6
	nx_subap = numpy.array([7]*n_wfs)
	n_subap = numpy.array([36]*n_wfs)

	pupil_mask = make_pupil_mask('circle', n_subap, nx_subap[0], 
			1., tel_diam)
	cus_pupilMask = pupil_mask.copy()
	cus_pupilMask[2] = 0
	cus_pupilMask[:,0] = 0
	hl_duds = pupil_mask + cus_pupilMask
	hl_duds_flat = hl_duds[pupil_mask==1].flatten()
	reduce_cents = numpy.tile(hl_duds_flat, 2*gs_pos.shape[0])
	n_subap = numpy.array([int(cus_pupilMask.sum())]*n_wfs)

	# onesMat, wfsMat_1, wfsMat_2, allMapPos, selector, xy_separations = roi_referenceArrays(
	# 	cus_pupilMask, gs_pos, tel_diam, roi_belowGround, roi_envelope)

	onesMat, wfsMat_1, wfsMat_2, allMapPos, selector, xy_separations = roi_referenceArrays(
				cus_pupilMask, gs_pos, tel_diam, roi_belowGround, roi_envelope)

	shwfs_centroids = fits.getdata('../../../../windProfiling/wind_paper/canary/data/test_fits/canary_noNoise_it10k_nl3_h0a10a20km_r00p1_L025_ws10a15a20_wd260a80a350_infScrn_wss448_gsPos0cn40a0c0a30c0.fits')#[:, :72*2]
	shwfs_centroids = shwfs_centroids[:, reduce_cents==2]

	covMapDim = 13
	roi_axis = 'x and y'
	mapping_type = 'mean'

	nr, nt = calculate_roi_covariance(shwfs_centroids, gs_pos, cus_pupilMask, tel_diam, roi_belowGround, roi_envelope, roi_axis, mapping_type)
	print('Time taken: {}'.format(nt))

	pyplot.figure()
	pyplot.imshow(nr)
