import matplotlib.pyplot as plt
from ipywidgets import interact
from modules.func import odd_even
from modules.CellLevel import spatial_map
import numpy as np

def trial_map(dfs):
    cells = dfs.columns[dfs.columns.str.contains('cell')]
    @interact(c=cells)
    def plot(c=cells[0]):
        plt.imshow(dfs.pivot_table(values=c,columns='bin',index='trial'))
        

def heat_map(dfs, cell_idx, align=None):
    #cell_idx: boolian index
    dfs2 = dfs_cells(dfs, cell_idx)
    sm = spatial_map(dfs2)
    sm /= sm.max(axis=0)
    if align is not None:
        idx = align
    else:
        idx = sm.idxmax(axis=0).sort_values().index
    sm = sm.loc[:, idx]
    sns.heatmap(sm.T)
    return idx


def rsa_matrix(dfs, cond1, cond2):
    fig, ax = plt.subplots(1,3,figsize=(20,8))
    dfs1, dfs2 = dfs.query(cond1), dfs.query(cond2)
    o1, e1 = odd_even(dfs1)
    o2, e2 = odd_even(dfs2)
    maps = []
    for i in [dfs1, dfs2, o1, e1, o2, e2]:
        maps.append(spatial_map(i))
    s = maps[0].shape[0]
    cors = (np.corrcoef(maps[0], maps[1])[:s, s:],\
            np.corrcoef(maps[2], maps[3])[:s, s:],\
            np.corrcoef(maps[4], maps[5])[:s, s:])
    print([c.shape for c in cors])
    mx = np.max(cors)
    mn = np.min(cors)
    titles = ["cond1 vs cond2", f"cond1 odd vs even", f"cond2 odd vs even"]
    fig.suptitle(f"cond1:{cond1} {len(dfs1.trial.unique())}trials, cond2:{cond2} {len(dfs2.trial.unique())}trials")
    for a, cor, title in zip(ax, cors, titles):
        im = a.imshow(cor, vmin=mn, vmax=mx, aspect='auto')
        a.set_title(title)
    plt.colorbar(im, ax=ax, location='bottom', aspect=50)