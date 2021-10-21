from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from multiprocessing import Pool, cpu_count
import tifffile as tf
import matplotlib.pyplot as plt
from pathlib import Path
from ipywidgets import interact
from importlib import reload
import sys
import seaborn as sns


def reloadf(func):
    reload(sys.modules[func.__module__])
    
def load_dfs(task, name, day, data_root):
    return pd.read_csv(data_root / task / name / day / "comb.csv")

def load_dfsdff(task, name, day, data_root):
    return pd.read_csv(data_root / task / name / day / "combdff.csv")

def ca_events(dfs, sigma=3):
    #sigma: of a gaussian filter to smoothen calcium traces before to detect calcium events.
    
    cell_col_names = dfs.loc[:,'cell0':].columns
    traces = dfs.groupby('trial').apply(lambda x: pd.DataFrame(gaussian_filter1d(x.loc[:,'cell0':], sigma=3, axis=0))).reset_index(drop=True)
    traces.columns = dfs.loc[:,'cell0':].columns
    dfs2 = dfs.iloc[:, :dfs.columns.get_loc('cell0')].join(traces)
    
    m = dfs2.loc[:,'cell0':].mean(axis=0)
    std = dfs2.loc[:,'cell0':].std(axis=0)
    def events(d):
        events = (d > m + 2*std) & (np.diff(d.loc[:,'cell0':], axis=0, prepend=np.inf) > 0)
        return events
    events = dfs2.groupby('trial').apply(lambda x: events(x.loc[:,'cell0':]))
    events = dfs2.iloc[:, :dfs.columns.get_loc('cell0')].join(events)
    return events
    

def smoothing(dfs, sigma=3):
    #smmothing calcium traces of a dfs along the time axis within each trial.
    cell_col_names = dfs.loc[:,'cell0':].columns
    traces = dfs.groupby('trial').apply(lambda x: pd.DataFrame(gaussian_filter1d(x.loc[:,'cell0':], sigma=3, axis=0))).reset_index(drop=True)
    traces.columns = dfs.loc[:,'cell0':].columns
    dfs2 = dfs.iloc[:, :dfs.columns.get_loc('cell0')].join(traces)   
    return dfs2


def parallel(func, args, n_process):
    p = Pool(n_process)
    ret = p.starmap(func, args)
    p.close()
    return ret


def odd_even(dfs):
    odd = dfs[dfs.trial.isin(dfs.trial.unique()[::2])]
    even = dfs[dfs.trial.isin(dfs.trial.unique()[1::2])]  
    return odd, even
