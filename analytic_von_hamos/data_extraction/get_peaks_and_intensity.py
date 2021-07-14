import numpy as np
from scipy.spatial.distance import cdist 
        ### cdist computes distance between each pair of the two collections of inputs. ###
from scipy.signal import find_peaks
        ###This function takes a one-dimensional array and finds all local maxima by simple comparison of neighbouring values. Optionally, a subset of these peaks can be selected by specifying conditions for a peak’s properties.###
from sklearn.cluster import DBSCAN #Density Based Spatial Clustering of Applications with Noise
        ### DBSCAN finds core samples of high density and expands clusters from them. Good for data which contains clusters of similar density.###
from sklearn.preprocessing import StandardScaler
        ### StandardScalar standardizes features by removing the mean and scaling to unit variance ###
import random


__all__ = ['reduce_to_peaks_with_intensities', 'partition_peaks_with_intensities']

#for smoothing an axis
def numpy_rbf(x,y,l): #generates array that apprx. a normalized radial basis function - 1's on the diagonal and values decreasing as they get further from the diagonal
    if x.ndim == 1:
        x = x[:,None]
    elif x.ndim == 2:
        pass
    else:
        raise ValueError('x should only have 2 dimensions')
    if y.ndim == 1:
        y = y[:,None]
    elif y.ndim == 2:
        pass
    else:
        raise ValueError('y should only have 2 dimensions')
    K = np.exp(-cdist(x,y,'sqeuclidean')/l**2) 
    # l is the smoothness factor - set to specific value in def reduce_to_peaks() 
    ### question for Franklin: what is the purpose of K? ###
    return K

def smoother(axlen,l): 
    K = numpy_rbf(np.linspace(-1,1,axlen),np.linspace(-1,1,axlen),l) #passes in two vectors with length of number of pixels in image dimensions and values evenly distributed between (-1,1)
    #linspace returns evenly spaced numbers over a specified interval.
    # axlen in the number of pixels in the image (passed to smoother in reduce_to_peaks)
    return K

def reduce_to_peaks_with_intensities(img,row_roi,col_roi, smoothness_factor=0.005, prominence=(0.005, 10), width=(10, 100)):
    """ We take an image and reduce it to a collection of peak points after smoothing it. We look for peaks along each column"""
    ## ROI = region of interest

    K = smoother(img.shape[0],smoothness_factor) ##returns square matrix with dim of img.shape[0] and 1's on the diagonal and covariances decreasing to 0 as values get further from diagonal
    K2 = smoother(img.shape[1],smoothness_factor) ## see above --but for other image dimension
    #img.shape returns a tuple of the number of rows, columns, and channels (if the image is color)

    t = np.zeros_like(img[:, 0]) #zeros_like return an arr ay of zeros with the same shape and type as a given array#
    
    simg = K @ img @ K2  ## simg = smoothed image , "@" indicates matrix multiplication #
    simg /= simg.max() #normalizes values of simg (all values now btwn 0 and 1)
    
    col_coords = []
    row_coords = []
    intensity_values = []

    for pixel in range(col_roi.start,col_roi.stop):
        peaks_data = find_peaks(simg[:,pixel],prominence=prominence,width=width) #finds peak in each column of images - prominence and width set the min values of peak width and relative peak height
        peaks = peaks_data[0]
        col_coords.extend(len(peaks)*[pixel]) #adds relevant pixel column number to col_coords?
        row_coords.extend(list(peaks)) #adds relevant pixel row coordinate to row_coords?
        for pk in range(len(peaks)):
            intensity = img[peaks[pk],[pixel]][0]
            intensity_values.append(intensity)
   
    p = np.stack((row_coords,col_coords,intensity_values))       
    return p #, surrounds


    

def partition_peaks_with_intensities(peaks_with_intensities,npeaks,eps=0.3,min_samples=10):
    #determine if x,y coordinates are row or column 
    if peaks_with_intensities.shape[0] == 3:
        peaks = peaks_with_intensities[:2,:]
        intensities = peaks_with_intensities[2,:]
        #convert int pixel values to floats and takes transpose
        #transpose of an array is non-contiguous -> make into contiguous array
        P = np.ascontiguousarray(peaks.astype('float64').T) 
        features = 0
    elif peaks_with_intensities.shape[1] == 3:
        peaks = peaks_with_intensities[:,:2]
        intensities = peaks_with_intensities[:,2]
        P = peaks.astype('float64')
        features = 1
    else:
        raise ValueError('expecting either 0 or 1st dim to have shape == 3')
    P = StandardScaler().fit_transform(P)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(P) #clusters points in P
    r = []
    for k in range(npeaks):
        if features == 0:
            peak_clusters_with_intensities = np.row_stack([peaks[:,db.labels_ == k],intensities[db.labels_ == k]])
            r.append(peak_clusters_with_intensities)
        else:
            peak_clusters_with_intensities = np.column_stack([peaks[db.labels_ == k, :],intensities[db.labels_ == k]])
            r.append(peak_clusters_with_intensities)
    return r

def get_surrounds(peaks):
    surr_x = []
    surr_y = []
    mk_surr_x = np.ones_like(peaks[0])   #scale this matrix to move surrounds closer/further from input pixels 
    surr_x.extend(np.add(peaks[0],mk_surr_x))
    surr_y.extend(peaks[1])
    surr_x.extend(np.subtract(peaks[0],mk_surr_x))
    surr_y.extend(peaks[1])
    return np.stack((surr_x,surr_y))


def get_zeros(data):

    assert data.shape[0] == 2 or data.shape[1] == 2, 'expecting 1st or 2nd dim to have shape == 2'
    if data.shape[0] == 2: 
        data = data.T
    pts_to_add = data.shape[0]*8 #adds 8 times the number of points in original dataset as random points with intensity zero
    existing_dataset = set(tuple(data[i,:]) for i in range(data.shape[0]))
    selected_x = np.random.randint(data[:,0].min(), data[:,0].max(),size=pts_to_add)
    selected_y = np.random.randint(data[:,1].min(), data[:,1].max(),size=pts_to_add)
    new_data = list(set(zip(selected_x,selected_y))) #making set first removes duplicates from new_data
    new_data = [(x,y) for (x,y) in new_data if not (x,y) in existing_dataset]
    new_data = np.array(new_data)   
    pts_overlapped = pts_to_add - new_data.shape[0]
    return new_data, pts_overlapped

def get_data_for_GPfit(detector_data):
    ####ADD assert here to check size, shape of detector_data

    detector_data = np.array([detector_data[1,:],detector_data[0,:]]) #detector pts orginially in form row0=horz pts, row1=vert pts
                                                                    #given we're about to transpose the data, we want row1=horz pts, row0=vert pts

    X_observed = detector_data.T
    # X_surrounds = get_surrounds(detector_data).T    #get_surrounds gets pixels left/right input data
   


    # X_random_pts, pts_overlapped = get_zeros(detector_data, 800) #get_zeros input: (dataset, #random pts to add)
    #                                         #get_zeros returns (pts, #of pts rejected due to overlap with existing data)
    X_random_pts, pts_overlapped = get_zeros(detector_data) #get_zeros input: (dataset, #random pts to add)
                                            #get_zeros returns (pts, #of pts rejected due to overlap with existing data)

    ##Create corresponding Y matrices for X_observed, X_surrounds, X_random_pts
    Y_ones_obs = np.ones((X_observed.shape[0],1))
    
    # Y_zeros_surr = np.zeros((X_surrounds.shape[0],1))
    

    Y_zeros_rand = np.zeros((X_random_pts.shape[0],1))

    # X = np.concatenate((X_observed, X_surrounds, X_random_pts))
    # Y = np.concatenate((Y_ones_obs, Y_zeros_surr, Y_zeros_rand))
    X = np.concatenate((X_observed, X_random_pts))
    Y = np.concatenate((Y_ones_obs, Y_zeros_rand))
    return X,Y

def get_data_for_GPfit_with_intensities(detector_data_with_intensities):
    ####ADD assert here to check size, shape of detector_data

    detector_data = np.array([detector_data_with_intensities[1,:],detector_data_with_intensities[0,:]]) #detector pts orginially in form row0=horz pts, row1=vert pts
                                                                    #given we're about to transpose the data, we want row1=horz pts, row0=vert pts
                                                                
    X_observed = detector_data.T
    # X_surrounds = get_surrounds(detector_data).T    #get_surrounds gets pixels left/right input data
   


    # X_random_pts, pts_overlapped = get_zeros(detector_data, 800) #get_zeros input: (dataset, #random pts to add)
    #                                         #get_zeros returns (pts, #of pts rejected due to overlap with existing data)
    X_random_pts, pts_overlapped = get_zeros(detector_data) #get_zeros input: (dataset, #random pts to add)
                                            #get_zeros returns (pts, #of pts rejected due to overlap with existing data)

    ##Create corresponding Y matrices for X_observed, X_surrounds, X_random_pts
    obs_intensities = detector_data_with_intensities[2,:]
    obs_intensities_max = np.max(obs_intensities)
    Y_observed = np.array(obs_intensities/obs_intensities_max).reshape(-1,1) #normalized intensities from detector
    
    # Y_zeros_surr = np.zeros((X_surrounds.shape[0],1))
    

    Y_zeros_rand = np.zeros((X_random_pts.shape[0],1))


    # X = np.concatenate((X_observed, X_surrounds, X_random_pts))
    # Y = np.concatenate((Y_ones_obs, Y_zeros_surr, Y_zeros_rand))
    X = np.concatenate((X_observed, X_random_pts))
    Y = np.concatenate((Y_observed, Y_zeros_rand))
    return X,Y
