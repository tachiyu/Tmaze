import seaborn as sns
from scipy.ndimage import gaussian_filter1d
import pandas as pd

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

def cellreg_id(task, name, day):
    d = pd.read_csv(fr"C:\Users\bdr\Tmaze_nVista\data\Tmaze\Tmaze nVista\processed_data\cellreg\{name}\cellreg.csv")
    p = pd.read_csv(fr"C:\Users\bdr\Tmaze_nVista\data\Tmaze\Tmaze nVista\processed_data\{task}\{name}\{day}\place_cell_properties.csv")
    ind = d[(d.Task==task)&(d.Name==name)&(d.Day==day)].index[0]
    d2 = d.iloc[ind, 3:]
    iis = []
    for i in range(p.shape[0]):
        idx = d2[d2==i].index
        if len(idx) > 0:
            iis.append(idx[0])
        else:
            iis.append(None)
            
    p['id'] = iis
    p.to_csv(fr"C:\Users\bdr\Tmaze_nVista\data\Tmaze\Tmaze nVista\processed_data\{task}\{name}\{day}\place_cell_properties.csv", index=False)
    

def load_cell_properties(task, name, day, pr):
    return pd.read_csv(pr/task/name/day/"place_cell_properties.csv")