import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import cpu_count, Pool


def shift(dfs):
    cell0_idx = dfs.columns.get_loc('cell0')
    cell_shuffled = pd.DataFrame(np.roll(dfs.loc[:,'cell0':], np.random.randint(0, dfs.shape[0]))
                                , columns = dfs.loc[:,'cell0':].columns)
    return dfs.iloc[:,:cell0_idx].reset_index(drop=True).join(cell_shuffled)


def nul_map(dfs, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return dfs.groupby('trial').apply(shift).reset_index(drop=True)


def null_maps(dfs, itr):
    p = Pool(cpu_count())
    args = [(dfs, i) for i in range(itr)]
    maps = p.starmap(nul_map, args)
    p.close()
    return maps


def spinfo(dfs, cond):
    dfs2 = dfs.query(cond)
    l = dfs2.loc[:,'cell0':].mean(axis=0)
    lx = dfs2.groupby('bin').mean().loc[:,'cell0':]
    px = dfs2.bin.value_counts(normalize=True)
    I = np.sum(lx.T*np.log2(lx/l).T*px, axis=1)
    return I
