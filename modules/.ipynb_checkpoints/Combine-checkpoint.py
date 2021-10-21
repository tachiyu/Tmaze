from modules.Caiman import load_cnm, load_Yr, detrend_df_f, cell_idx
from pathlib import Path
import pandas as pd
data_root = r"C:\Users\bdr\Tmaze_nVista\data\Tmaze\Tmaze nVista\processed_data"

def comb_to_combdff(task, name, day):
    Yr = load_Yr(task, name, day)
    cnm = load_cnm(task, name, day)
    A = cnm.estimates.A
    cnm.estimates.f = None
    B = cnm.estimates.compute_background(Yr[0])
    C = cnm.estimates.C
    YrA = cnm.estimates.YrA
    
    DFF = detrend_df_f(A, B, C, YrA, quantileMin=10, frames_window=500, 
                 flag_auto=False, use_fast=True, detrend_only=False)
    cell_idxs = cell_idx(cnm)
    DFF = pd.DataFrame(DFF[cell_idxs, :].T, columns=[f'cell{n}' for n in range(len(cell_idxs))])
    
    comb = pd.read_csv(Path(data_root)/task/name/day/"comb.csv")
    cell0_idx = comb.columns.get_loc('cell0')
    combdff = comb.iloc[:, :cell0_idx].join(DFF)
    
    combdff.to_csv(Path(data_root)/task/name/day/"combdff.csv", index=False)
    