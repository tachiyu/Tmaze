import warnings
warnings.simplefilter('ignore', FutureWarning) #to temporally ignore annoying future warning with importing caiman.
import caiman as cm
warnings.resetwarnings()
import scipy
from pathlib import Path
from caiman.utils.stats import df_percentile
from caiman.source_extraction.cnmf.utilities import fast_prct_filt
import numpy as np

def load_cnm(task,name,day,data_root):
    return cm.source_extraction.cnmf.cnmf.load_CNMF(Path(data_root)/task/name/day/"cnmp0noisx.hdf5")

def load_Yr(task,name,day,data_root):
    Yr, dims, T = cm.mmapping.load_memmap(list((Path(data_root)/task/name/day).glob('*.mmap'))[0])
    return Yr, dims, T

def load_fp(task, name, day,data_root):
    cnm = load_cnm(task,name,day,data_root)
    return np.array(cnm.estimates.A.todense()).reshape((270,360,-1), order='F')\
            .transpose((2,0,1))[cell_idx(cnm)]

def cell_idx(cnm):
    #cut off upper cell (above 30 pixel)
    A = np.array(cnm.estimates.A.todense()).reshape((270,360,-1), order='F')
    idx1 = []
    for i in range(A.shape[2]):
        if (A[:30, :, i] == 0).all():
            idx1.append(i)
    #good components
    idx2 = cnm.estimates.idx_components
    
    idx = np.intersect1d(idx1, idx2)
    return idx
    
def detrend_df_f(A, B, C, YrA, quantileMin=8, frames_window=500, 
                 flag_auto=True, use_fast=False, detrend_only=False):
    """ Compute DF/F signal without using the original data.
    In general much faster than extract_DF_F. *modulated from the original sorce code by Yuto Tachiki*

    Args:
        A: scipy.sparse.csc_matrix
            spatial components (from cnmf cnm.A)

        B: ndarray
            background components

        C: ndarray
            temporal components (from cnmf cnm.C)
            
        YrA: ndarray
            residual signals

        quantile_min: float
            quantile used to estimate the baseline (values in [0,100])
            used only if 'flag_auto' is False, i.e. ignored by default

        frames_window: int
            number of frames for computing running quantile

        flag_auto: bool
            flag for determining quantile automatically

        use_fast: bool
            flag for using approximate fast percentile filtering

        detrend_only: bool (False)
            flag for only subtracting baseline and not normalizing by it.
            Used in 1p data processing where baseline fluorescence cannot be
            determined.

    Returns:
        F_df:
            the computed Calcium activity to the derivative of f
    """

    nA = np.sqrt(np.ravel(A.power(2).sum(axis=0)))
    nA_mat = scipy.sparse.spdiags(nA, 0, nA.shape[0], nA.shape[0])
    nA_inv_mat = scipy.sparse.spdiags(1. / nA, 0, nA.shape[0], nA.shape[0])
    A = A * nA_inv_mat
    C = nA_mat * C
    if YrA is not None:
        YrA = nA_mat * YrA

    F = C + YrA if YrA is not None else C
    B = A.T.dot(B)
    T = C.shape[-1]

    if flag_auto:
        data_prct, val = df_percentile(F[:, :frames_window], axis=1)
        if frames_window is None or frames_window > T:
            Fd = np.stack([np.percentile(f, prctileMin) for f, prctileMin in
                           zip(F, data_prct)])
            Df = np.stack([np.percentile(f, prctileMin) for f, prctileMin in
                           zip(B, data_prct)])
            if not detrend_only:
                F_df = (F - Fd[:, None]) / (Df[:, None] + Fd[:, None])
            else:
                F_df = F - Fd[:, None]
        else:
            if use_fast:
                Fd = np.stack([fast_prct_filt(f, level=prctileMin,
                                              frames_window=frames_window) for
                               f, prctileMin in zip(F, data_prct)])
                Df = np.stack([fast_prct_filt(f, level=prctileMin,
                                              frames_window=frames_window) for
                               f, prctileMin in zip(B, data_prct)])
            else:
                Fd = np.stack([scipy.ndimage.percentile_filter(
                    f, prctileMin, (frames_window)) for f, prctileMin in
                    zip(F, data_prct)])
                Df = np.stack([scipy.ndimage.percentile_filter(
                    f, prctileMin, (frames_window)) for f, prctileMin in
                    zip(B, data_prct)])
            if not detrend_only:
                F_df = (F - Fd) / (Df + Fd)
            else:
                F_df = F - Fd
    else:
        if frames_window is None or frames_window > T:
            Fd = np.percentile(F, quantileMin, axis=1)
            Df = np.percentile(B, quantileMin, axis=1)
            if not detrend_only:
                F_df = (F - Fd[:, None]) / (Df[:, None] + Fd[:, None])
            else:
                F_df = F - Fd[:, None]
        else:
            Fd = scipy.ndimage.percentile_filter(
                F, quantileMin, (1, frames_window))
            Df = scipy.ndimage.percentile_filter(
                B, quantileMin, (1, frames_window))
            if not detrend_only:
                F_df = (F - Fd) / (Df + Fd)
            else:
                F_df = F - Fd

    return F_df