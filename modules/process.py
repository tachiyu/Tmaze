import numpy as np
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET
import tifffile as tf
import caiman as cm
from caiman.source_extraction import cnmf
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params
from modules.cell import cell_idx, detrend_df_f

def frames(imaging_root, processed_root, task="", name="", day=""):
    raw_movie_path = list((Path(imaging_root)/name/task/day).glob('rec*xml'))
    frames = []
    for p in raw_movie_path:
        root = ET.parse(p).getroot()
        for chd in root[0]:
            if chd.items()[0][1] == "dropped_count":
                if int(chd.text) > 0:
                    frames.append(None)
                    break
            if chd.items()[0][1] == "frames":
                frames.append(int(chd.text))
                break
    (Path(processed_root)/name/task/day).mkdir(parents=True, exist_ok=True)
    np.savetxt(Path(processed_root)/name/task/day/"frames.txt", frames, fmt='%0i')
    

def movie_concat(imaging_root, processed_root, task="", name="", day=""):
    raw_movie_path = list((Path(imaging_root)/name/task/day).glob('rec*xml'))
    frames = np.loadtxt(Path(processed_root)/name/task/day/'frames.txt')
    mvs = []
    for p,frame in zip(raw_movie_path,frames):
        if frame:
            root = ET.parse(p).getroot()
            tif_path = root[1][0].text
            mvs.append(tf.imread(Path(imaging_root)/name/task/day/tif_path))
    mv = np.concatenate(mvs)
    tf.imsave(Path(processed_root)/name/task/day/"mv.tif", data=mv)
    
    
def motion_correct(processed_root, task="", name="", day="", dview=None, n_processes=1, frate=20, **kwargs):
    fnames = list((Path(processed_root)/name/task/day).glob("*.tif"))
    # dataset dependent parameters
    decay_time = 0.4                 # length of a typical transient in seconds

    # motion correction parameters
    pw_rigid = False         # flag for performing piecewise-rigid motion correction (otherwise just rigid)
    gSig_filt = (3, 3)       # size of high pass spatial filtering, used in 1p data
    max_shifts = (5, 5)      # maximum allowed rigid shift
    strides = (48, 48)       # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24)      # overlap between pathes (size of patch strides+overlaps)
    max_deviation_rigid = 3  # maximum deviation allowed for patch with respect to rigid shifts
    border_nan = 'copy'      # replicate values along the boundaries

    mc_dict = {
        'fr': frate,
        'decay_time': decay_time,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'gSig_filt': gSig_filt,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': border_nan
    }
    bord_px = 0
    opts = params.CNMFParams(params_dict=mc_dict)
    if len(kwargs) > 0:
        opts.change_params(params_dict=kwargs)
        
    if motion_correct:
        # do motion correction rigid
        mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
        mc.motion_correct(save_movie=True)
        fname_mc = mc.fname_tot_els if pw_rigid else mc.fname_tot_rig
        if pw_rigid:
            bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                         np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
        else:
            bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(np.int)

        bord_px = 0 if border_nan is 'copy' else bord_px
        fname_new = cm.save_memmap(fname_mc, base_name='memmap_', order='C',
                                   border_to_0=bord_px)
    else:  # if no motion correction just memory map the file
        fname_new = cm.save_memmap(fnames, base_name='memmap_',
                                   order='C', border_to_0=0, dview=dview)
    Path(list((Path(processed_root)/name/task/day).glob("*rig*mmap"))[0]).unlink()
    
    
def cnmfe(processed_root, task="", name="", day="", dview=None, n_processes=1, frate=20, **kwargs):
    fname_new = list((Path(processed_root)/name/task/day).glob("*memmap*"))[0]

    # dataset dependent parameters
    decay_time = 0.4                 # length of a typical transient in seconds

    mc_dict = {
        'fr': frate,
        'decay_time': decay_time,
    }
    bord_px = 0
    opts = params.CNMFParams(params_dict=mc_dict)

    # parameters for source extraction and deconvolution
    p = 0               # order of AR model
    K = None            # upper bound on number of components per patch, in general None
    gSig = (5, 5)       # gaussian width of a 2D gaussian kernel, which approximates a neuron
    gSiz = (11, 11)     # average diameter of a neuron, in general 4*gSig+1
    Ain = None          # possibility to seed with predetermined binary masks
    merge_thr = .7      # merging threshold, max correlation allowed
    rf = None             # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
    stride_cnmf = 20    # amount of overlap between the patches in pixels
    #                     (keep it at least large as gSiz, i.e 4 times the neuron size gSig)
    tsub = 2            # downsampling factor in time for initialization,
    #                     increase if you have memory problems
    ssub = 1            # downsampling factor in space for initialization,
    #                     increase if you have memory problems
    #                     you can pass them here as boolean vectors
    low_rank_background = None  # None leaves background of each patch intact,
    #                     True performs global low-rank approximation if gnb>0
    gnb = 0             # number of background components (rank) if positive,
    #                     else exact ring model with following settings
    #                         gnb= 0: Return background as b and W
    #                         gnb=-1: Return full rank background B
    #                         gnb<-1: Don't return background
    nb_patch = 0        # number of background components (rank) per patch if gnb>0,
    #                     else it is set automatically
    min_corr = .8       # min peak value from correlation image
    min_pnr = 10        # min peak to noise ration from PNR image
    ssub_B = 2          # additional downsampling factor in space for background
    ring_size_factor = 1.4  # radius of ring is gSiz*ring_size_factor

    opts.change_params(params_dict={'method_init': 'corr_pnr',  # use this for 1 photon
                                    'K': K,
                                    'gSig': gSig,
                                    'gSiz': gSiz,
                                    'merge_thr': merge_thr,
                                    'p': p,
                                    'tsub': tsub,
                                    'ssub': ssub,
                                    'rf': rf,
                                    'stride': stride_cnmf,
                                    'only_init': True,    # set it to True to run CNMF-E
                                    'nb': gnb,
                                    'nb_patch': nb_patch,
                                    'method_deconvolution': 'oasis',       # could use 'cvxpy' alternatively
                                    'low_rank_background': low_rank_background,
                                    'update_background_components': True,  # sometimes setting to False improve the results
                                    'min_corr': min_corr,
                                    'min_pnr': min_pnr,
                                    'normalize_init': False,               # just leave as is
                                    'center_psf': True,                    # leave as is for 1 photon
                                    'ssub_B': ssub_B,
                                    'ring_size_factor': ring_size_factor,
                                    'del_duplicates': True,                # whether to remove duplicates from initialization
                                    'border_pix': bord_px})                # number of pixels to not consider in the borders)
    if len(kwargs) > 0:
        opts.change_params(params_dict=kwargs)
            
    # load memory mappable file
    Yr, dims, T = cm.load_memmap(fname_new)
    images = Yr.T.reshape((T,) + dims, order='F')
    
    cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=Ain, params=opts)
    cnm.fit(images)
    
    #%% COMPONENT EVALUATION
    # the components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier

    min_SNR = 3            # adaptive way to set threshold on the transient size
    r_values_min = 0.85    # threshold on space consistency (if you lower more components
    #                        will be accepted, potentially with worst quality)
    cnm.params.set('quality', {'min_SNR': min_SNR,
                               'rval_thr': r_values_min,
                               'use_cnn': False})
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

    print(' ***** ')
    print('Number of total components: ', len(cnm.estimates.C))
    print('Number of accepted components: ', len(cnm.estimates.idx_components))
    
    cnm.save(str(Path(processed_root)/name/task/day/"cnm.hdf5"))
    

def bdata_process(raw_behavior_root, processed_root, task="", name="", day=""):
    #original behavior data
    filename = list((Path(raw_behavior_root)/name/task/day).glob('*.csv'))[0]
    print(f"start processing {filename}")
    bdata = pd.read_csv(filename, delimiter=';')

    #timepoints of starts of trials
    tristarts = np.where(bdata.yPosComp.diff() < -500)[0]
    
    #a timepoint where the first trial started
    expstart =  np.where(bdata.yPosComp.diff() < -3000)[0]
    if len(expstart) != 1:
        print(f"error!, {name},{task},{day}")

    #cut off unnecessary timepoints
    bdata2 = bdata.iloc[tristarts[0]:tristarts[-1],:]

    #make behavioral variables for each timepoint
    t = (bdata2.DateTime - bdata2.DateTime.iloc[0]) * 1000 * 60 * 60 * 24 #time(ms)
    trials = [] #trial number of current timepoint
    typs = [] #trial type ("AA",..,"BA") of current timepoint
    choices = [] #choice of the trial of current timepoint
    rewardeds = [] #1 if reward was derivered in the trial of current timepoint, else 0
    pretyps = [] #trial type ("AA",..,"BA") of the previous trial of current timepoint
    prechoices = [] #choice of the previous trial of current timepoint
    prerewardeds = [] #1 if reward was derivered in the previous trial of current timepoint, else 0
    bins = [] #spatial bin of current timepoint
    speeds = [] #speed
    rundirs = [] #rundir(differences of yPos between adjacent timepoints)
                 #rundirs<0 means left turn and rundirs>0 means right turn
    for i in range(len(tristarts)-1):
        if i == 0:
            pretyps += [None] * (tristarts[i+1] - tristarts[i])
            prechoices += [None] * (tristarts[i+1] - tristarts[i])
            prerewardeds += [None] * (tristarts[i+1] - tristarts[i])
        else:
            pretyps += [typ] * (tristarts[i+1] - tristarts[i])
            prechoices += [choice] * (tristarts[i+1] - tristarts[i])
            prerewardeds += [rewarded] * (tristarts[i+1] - tristarts[i])
        start = bdata2.loc[tristarts[i],:]
        end = bdata2.loc[tristarts[i+1]-1,:]
        choice = 'L' if end.ViewDir > 180 else 'R'
        type1 = 'A' if start.xPosComp > 0 else 'B'
        type2 = 'A' if end.xPosComp > 0 else 'B'
        typ = type1 + type2
        rewarded = 1 if (choice, type2) in (('L','A'),('R','B')) else 0
        trials += [i] * (tristarts[i+1] - tristarts[i])
        typs += [typ] * (tristarts[i+1] - tristarts[i])
        choices += [choice] * (tristarts[i+1] - tristarts[i])    
        rewardeds += [rewarded] * (tristarts[i+1] - tristarts[i])
        gbdata = bdata2.loc[tristarts[i]:tristarts[i+1]-1,:]
        viewdir = np.where(gbdata.ViewDir < 180, gbdata.ViewDir, gbdata.ViewDir-360)
        tjuncX = gbdata.iloc[np.where(gbdata.ViewDir==0)[0][-1]].xPosComp
        bin1 = np.where(abs(viewdir) < 90, np.clip(gbdata.yPosComp, 0, 1000)//50, 0)
        bin2 = np.where(abs(viewdir) == 90, np.clip(abs(gbdata.xPosComp - tjuncX), 0, 499)//50 + 21, 0)
        bins += list(bin1+bin2)
        rundirs += list(np.diff(gbdata.yPos, prepend=gbdata.yPos.iloc[0]))
        speeds += list(np.sqrt(np.diff(gbdata.xPos, prepend=gbdata.xPos.iloc[0])**2 \
                               + np.diff(gbdata.yPos, prepend=gbdata.yPos.iloc[0])**2))

    #convert 0 < viewdir < 360 to -90 < viewdir < 90 (counterclockwise)
    viewdir = np.where(bdata2.ViewDir < 180, bdata2.ViewDir, bdata2.ViewDir-360)

    #True if the current trial was imaged.
    imaging = np.where(abs(bdata2.xPosComp) > 2000, True, False)

    #visual stimuli of current timepoint
    stimulus = np.where(bdata2.xPosComp > 0, "A", "B")

    bdata3 = pd.DataFrame({'t': t, 
                           'trial' : trials,
                           'bin' : bins,
                           'type' : typs,
                           'choice' : choices,
                           'rewarded' : rewardeds,
                           'rundir' : rundirs,
                           'speed' :speeds,
                           'stimulus' : stimulus,
                           'pretype' : pretyps,
                           'prechoice' : prechoices,
                           'prerewarded' : prerewardeds,
                           'x' : bdata2.xPosComp,
                           'y' : bdata2.yPosComp,
                           'viewdir' : viewdir,
                           'xreal' : bdata2.xPos,
                           'yreal' : bdata2.yPos,
                           'isimaged' : imaging})
    
    bdata3.to_csv(Path(processed_root)/name/task/day/"bdata.csv", index=False)
    return bdata3


def comb_for_trial_recording(processed_root, task="", name="", day=""):
    # loading data
    bdata = pd.read_csv(Path(processed_root)/name/task/day/"bdata.csv")
    frames = np.loadtxt(Path(processed_root)/name/task/day/"frames.txt")
    cnm = cnmf.cnmf.load_CNMF(Path(processed_root)/name/task/day/"cnm.hdf5", n_processes=1, dview=None)

    # use only imaging data
    bdata2 = bdata[bdata.isimaged]

    traces = cnm.estimates.C[cell_idx(cnm)].T
    traces = pd.DataFrame(traces, columns=[f'cell{i}' for i in range(traces.shape[1])])

    #some variables for traces
    f = []
    for i in range(len(frames)):
        f += [i]*int(frames[i])
    t = []
    for i in range(len(frames)):
        t += [50.*j for j in range(int(frames[i]))]
    traces['t'] = t
    traces['trial'] = f

    if len(traces.trial.unique()) != len(bdata2.trial.unique()):
        print(f'error!{task}{name}{day}')

    # merge traces and bdata, making combs
    combs = []
    for tr, bd in zip(traces.groupby('trial'), bdata2.groupby('trial')):
        bd[1]['t'] -= bd[1]['t'].iloc[0]
        combs.append(pd.merge_asof(tr[1].loc[:,:'t'], bd[1], on='t'))
    comb = pd.concat(combs).reset_index(drop=True)

    # finally, arign colums
    t_idx = comb.columns.get_loc('t')
    cols = list(comb.columns[t_idx:]) + list(comb.columns[:t_idx])
    comb = comb[cols]
    
    comb.to_csv(Path(processed_root)/name/task/day/"comb.csv", index=False)
    return comb

   
#for block_imaging records.
def comb_for_block_recording(processed_root, task="", name="", day=""):
    # loading data
    bdata = pd.read_csv(Path(processed_root)/name/task/day/"bdata.csv")
    frames = np.loadtxt(Path(processed_root)/name/task/day/"frames.txt")
    cnm = cnmf.cnmf.load_CNMF(Path(processed_root)/name/task/day/"cnm.hdf5", n_processes=1, dview=None)

    # use only imaging data
    bd = bdata[bdata.isimaged]
    imidx = bd.trial.unique()

    traces = cnm.estimates.C[cell_idx(cnm)].T
    traces = pd.DataFrame(traces, columns=[f'cell{i}' for i in range(traces.shape[1])])

    #divide bdata into blocks
    blocks = []
    block = []
    for i in range(imidx.max()+1):
        if i not in imidx:
            bd_block = bd[bd.trial.isin(block)]
            bd_block['t'] -= bd_block['t'].iloc[0]
            blocks.append(bd_block)
            block = []
        else:
            block.append(i)
    bd_block = bd[bd.trial.isin(block)]
    bd_block['t'] -= bd_block['t'].iloc[0]
    blocks.append(bd_block)
    
    #divide traces into blocks
    block_frames = []
    t = []
    for i in range(len(frames)):
        block_frames += [i]*int(frames[i])
        t += [50.*j for j in range(int(frames[i]))]
    traces['block'] = block_frames
    traces['t'] = t

    if len(traces.block.unique()) != len(blocks):
        print(f'error!{task}{name}{day}')

    # merge traces and bdata, making combs
    combs = []
    for tr, bd in zip(traces.groupby('block'), blocks):
        combs.append(pd.merge_asof(tr[1].loc[:,:'t'], bd, on='t', direction='nearest'))
    comb = pd.concat(combs).reset_index(drop=True)

    # finally, arign colums
    t_idx = comb.columns.get_loc('t')
    cols = list(comb.columns[t_idx:]) + list(comb.columns[:t_idx])
    comb = comb[cols]
    
    comb.to_csv(Path(processed_root)/name/task/day/"comb.csv", index=False)
    return comb


def comb_to_combdff(processed_root, task="", name="", day=""):
    Yr, dims, T = cm.mmapping.load_memmap(list((Path(processed_root) / name / task / day).glob('*.mmap'))[0])
    cnm = cnmf.cnmf.load_CNMF(Path(processed_root)/name/task/day/"cnm.hdf5", n_processes=1, dview=None)
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
