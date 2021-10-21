import sys
sys.path.append(r"C:\Users\bdr\Tmaze")
from modules import *

proj_name = "Reversal_nVista2"
pt = Path(fr"C:\Users\bdr\Tmaze\project\{proj_name}\data\project_table.csv")
pc = Path(fr"C:\Users\bdr\Tmaze\project\{proj_name}\data\processed_data")
ri = Path(fr"C:\Users\bdr\Tmaze\project\{proj_name}\data\raw_imaging_data")
rb = Path(fr"C:\Users\bdr\Tmaze\project\{proj_name}\data\raw_behavior_data")

def load_dfs(task, name, day):
    return pd.read_csv(pc  / name / task / day / "comb.csv")

def load_dfsdff(task, name, day, data_root):
    return pd.read_csv(pc / name / task / day / "combdff.csv")

def load_cnm(task,name,day):
    return cm.source_extraction.cnmf.cnmf.load_CNMF(Path(pc) / name / task / day / "cnm.hdf5")

def load_Yr(task,name,day):
    Yr, dims, T = cm.mmapping.load_memmap(list((Path(pc) / name / task / day).glob('*.mmap'))[0])
    return Yr, dims, T

def load_fp(task, name, day):
    cnm = load_cnm(task,name,day)
    return np.array(cnm.estimates.A.todense()).reshape((270,360,-1), order='F')\
            .transpose((2,0,1))[cell_idx(cnm)]

def load_bdata(task, name, day):
    return pd.read_csv(pc  / name / task / day / "bdata.csv")

def load_frames(task, name, day):
    return np.loadtxt(pc  / name / task / day / "frames.txt")

def parallel(func, args, n_processes=None):
    if not n_processes:
        p = Pool()
    else:
        p = Pool(n_processes)
    r = p.starmap(func, args)
    p.close()
    return r