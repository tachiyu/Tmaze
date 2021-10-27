import seaborn as sns
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import numpy as np
from pathlib import Path
import re

def dfs_cells(dfs, bool_idx):
    dfs2 = dfs.drop(columns=[f'cell{n}' for n in range(len(bool_idx)) if not bool_idx[n]])
    return dfs2
    
def spatial_map(dfs):
    d = dfs.groupby(['bin']).mean().reset_index()
    for i in range(31):
        if i not in d.index:
            d.loc[i] = None
            d.loc[i,'bin'] = i
    d = d.interpolate()
    cells = gaussian_filter1d(d.loc[:, d.columns.str.contains('cell')], sigma=1, axis=0)
    cells = pd.DataFrame(cells, index=range(cells.shape[0]), columns=d.columns[d.columns.str.contains('cell')])
    return cells

def cellreg_id(task, name, day, pc, file_title):
    #add 'id' columns to 'place_cell_properties.csv'. 'id' columns shows global id (via cellreg) of each cells in that day.
    print(task,name,day)
    d = pd.read_csv(Path(pc)/"cellreg"/name/"cellreg.csv")
    p = pd.read_csv(Path(pc)/task/name/day/file_title)
    ind = d[(d.Task==task)&(d.Name==name)&(d.Day==day)].index[0]
    d2 = d.iloc[ind, 3:]
    iis = []
    for i in range(p.shape[0]):
        idx = d2[d2==i].index
        if len(idx) > 0:
            iis.append(re.sub(r"\D", "", idx[0]))
        else:
            iis.append(None)

    if 'id' in p.columns:
        p['id'] = iis
    else:
        p.insert(0,'id',iis)
    p.to_csv(Path(pc)/task/name/day/file_title, index=False)
    

def load_cell_properties(task, name, day, pr):
    return pd.read_csv(pr/task/name/day/"place_cell_properties.csv")


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