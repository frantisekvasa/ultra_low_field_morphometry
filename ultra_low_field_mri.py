
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:11:24 2023

@author: Frantisek Vasa (frantisek.vasa@kcl.ac.uk)

Analysis of data from 23 healthy adult participants, scanned on two 64 mT Hyperfine Swoop MRI systems, and a 3 T GE Premier MRI system.
Segmentation, parcellation and volume estimation was carried out using SynthSeg+, as implemented in FreeSurfer 7.3.2.
"""

# %% libraries etc

# general
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import dice
import math
import pingouin as pg

# plotting / tables
from matplotlib import pyplot as plt
from matplotlib import colors
import seaborn as sb
import csv

# neuroimaging
import nibabel as nib

# home directory
import os
from pathlib import Path
home_dir = str(Path.home()) # home directory
hype_dir = home_dir+'/Data/HYPE'

# directory to save figures
plot_dir = home_dir+'/Desktop/HYPE_plots'
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)

# flag to determine whether figures are saved
save_fig = True

# change plot font
plt.rcParams["font.family"] = "arial"

# plotting parameters
# general
lbs = 22 # label size
lgs = 20 # legend size
axs = 18 # axis size

# %%

"""
functions
"""

# %% Lin'c Concordance Correlation Coefficient (CCC)

# https://github.com/murraylab/spatiotemporal/blob/main/spatiotemporal/extras.py

def lin(x, y):
    """Lin's concordance correlation coefficient

    Lin's concordnce ranges from -1 to 1 and achieves a tradeoff between
    correlation and variance explained.

    Args:
      x, y (lists or 1xN numpy arrays): the two vectors on which to find Lin's concordance

    Returns:
      float: The Lin's concordance of the two given vectors
    """
    return 2*np.cov(x, y, ddof=0)[0,1]/(np.var(x) + np.var(y) + (np.mean(x)-np.mean(y))**2)

# %% Rounding (Celing / Floor) to a fixed number of decimal places (for plot colorbar limits)

# https://stackoverflow.com/questions/58065055/floor-and-ceil-with-number-of-decimals

def deci_ceil(a, precision=0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)

def deci_floor(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)

# %% plotting anatomical scans and segmentations

from matplotlib import cm
import nilearn as nl
from nilearn import plotting

def plot_nl_image_masked(img_vec,mask_vec,img_shape,img_affine,cmap,clim=None,*line_args,**line_kwargs):
    if clim is None:
        #clim = (min(img_vec[mask_vec==1]),max(img_vec[mask_vec==1]))
        clim = (min(img_vec[mask_vec==1]),np.percentile(img_vec[mask_vec==1],95))
    # i) edit image and colorbar to map background to black
    img_masked = np.ones(img_vec.size)*(clim[0]-1); img_masked[mask_vec==1] = img_vec[mask_vec==1]
    cmap_under = cm.get_cmap(cmap, 256); cmap_under.set_under('white')
    # ii) convert image to nii and plot
    img_masked_nii = nib.Nifti1Image(np.reshape(img_masked,img_shape),affine=img_affine)
    nl.plotting.plot_img(img_masked_nii,colorbar=True,cmap=cmap_under, vmin=clim[0], vmax=clim[1],*line_args,**line_kwargs)

def plot_nl_seg_masked(img_vec,mask_vec,img_shape,img_affine,cmap,clim=None,*line_args,**line_kwargs):
    if clim is None:
        #clim = (min(img_vec[mask_vec==1]),max(img_vec[mask_vec==1]))
        #clim = (min(img_vec[mask_vec==1]),np.percentile(img_vec[mask_vec==1],95))
        clim = (min(img_vec),max(img_vec))
    # i) edit image and colorbar to map background to black
    #img_masked = np.ones(img_vec.size)*(clim[0]-1); img_masked[mask_vec==1] = img_vec[mask_vec==1]
    #cmap_under = cm.get_cmap(cmap, 256); cmap_under.set_under('white')
    # ii) convert image to nii and plot
    img_masked_nii = nib.Nifti1Image(np.reshape(img_vec,img_shape),affine=img_affine)
    nl.plotting.plot_img(img_masked_nii,colorbar=False,cmap=cmap, vmin=clim[0], vmax=clim[1],*line_args,**line_kwargs)

# %% correct SynthSeg csv volumes

# correct local "ctx" tissue volumes
# (only necessary if SynthSeg was run before 17.07.2024; see https://github.com/BBillot/SynthSeg/commits/master/SynthSeg/predict_synthseg.py)

# Instructions from Benjamin Billot:
    
# To correct the "ctx" volumes without re-running SynthSeg, let's define:
# ctx_volumes_old = a vector with the volumes of all 68 "ctx" labels given by the FAULTY SynthSeg version
# ctx_rh_volumes_new = a vector with the volumes of all 34 "ctx-rh" RIGHT labels given by the NEW SynthSeg version
# ctx_lh_volumes_new = a vector with the volumes of all 34 "ctx-lh" LEFT labels given by the NEW SynthSeg version
# wm_volume = the total volume of white matter (both hemishperes)
# left_cortex_volume = the volume for the label "left cerebral cortex"
# right_cortex_volume = the volume for the label "right cerebral cortex"

# First, we need to "de-normalise" the white matter
# ctx_volumes_tmp = ctx_volumes_old / wm_volume

# Now we split ctx_volumes_tmp in two, which gives us ctx_lh_volumes_tmp and ctx_rh_volumes_tmp. To get the final ctx volumes, we simply do:
# ctx_lh_volumes_new = ctx_lh_volumes_tmp / sum(ctx_lh_volumes_tmp) * left_cortex_volume
# ctx_rh_volumes_new = ctx_rh_volumes_tmp / sum(ctx_rh_volumes_tmp) * right_cortex_volume

def correct_synthseg_csv(csv_in,csv_out):
    
    """Correct SynthSeg cortical parcellation "ctx" volumes
    
    Only necessary if SynthSeg was run before 17.07.2024
    see: https://github.com/BBillot/SynthSeg/commits/master/SynthSeg/predict_synthseg.py

    Args:
      csv_in: path to and name of input .csv file (generated by SynthSeg)
      csv_out: path to and name of output corrected .csv file
    """
    
    # does input file exist?
    if not (os.path.isfile(csv_in)):
        raise Exception('input csv file does not exist')
    
    ### volume indices, based on N=101 "SynthSeg --parc" numerical outputs (i.e. excluding filename)
    
    # left / right cerebral cortex
    cortex_l_ind = [2]
    cortex_r_ind = [20]
    cortex_ind = cortex_l_ind + cortex_r_ind
    
    # white matter
    wm_ind = [1,19]
    
    # "ctx" cortical parcels
    # [i for i, elem in enumerate(vol) if elem in gmc_nm]
    ctx_l_ind = [33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66]
    ctx_r_ind = [67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100]
    ctx_ind = ctx_l_ind + ctx_r_ind
    
    # N ctx per hemisphere
    n_ctx_h = len(ctx_l_ind)
    
    ### load file
    pd_in = pd.read_csv(csv_in)
    vol_in = np.array(pd_in.iloc[0,1:],dtype=float)
    
    ### diagnosis: is sum(left & right cerebral cortex) = sum(ctx), in the *input*?
    if np.abs(np.sum(vol_in[ctx_ind]) - np.sum(vol_in[cortex_ind])) > 1:
        print('--\nctx volume correction is required')

        # (pre-)create output
        pd_out = pd_in.copy()
        vol_out = np.copy(vol_in)

        # "de-normalise" the white matter
        ctx_vol_tmp = vol_in[ctx_ind] / np.sum(vol_in[wm_ind])
        
        # split ctx_volumes_tmp in two and "re-normalise"
        ctx_l_vol_out = ctx_vol_tmp[:n_ctx_h] / np.sum(ctx_vol_tmp[:n_ctx_h]) * vol_in[cortex_l_ind]
        ctx_r_vol_out = ctx_vol_tmp[n_ctx_h:] / np.sum(ctx_vol_tmp[n_ctx_h:]) * vol_in[cortex_r_ind]
        
        # update volumes in vol_out
        vol_out[ctx_l_ind] = ctx_l_vol_out
        vol_out[ctx_r_ind] = ctx_r_vol_out
        
        # quality check: is sum(left & right cerebral cortex) = sum(ctx), in the *output*?
        if np.abs(np.sum(vol_out[ctx_ind]) - np.sum(vol_in[cortex_ind])) > 1:
            print('ctx volumes not appropriately corrected')
        else:
            print('ctx volumes corrected')
        
        # update pd_out
        pd_out.iloc[0,1:] = vol_out
        pd_out.rename(columns={'Unnamed: 0': ''}, inplace=True)
        
        # write csv_out
        pd_out.to_csv(csv_out, index=False, float_format='%.3f') # quoting=csv.QUOTE_NONNUMERIC
        
        if (os.path.isfile(csv_out)):
            print('output csv file written\n--')
        
    else:
        print('--\nno correction is required\n--')

# %%

"""
demographics and set-up
"""

# %% demographics

part = pd.read_csv(hype_dir+'/participants.tsv', sep='\t')

sub = np.array(part['participant_id'],dtype='str')
age = np.array(part['age'])
sex = np.array(part['sex'],dtype='str')

ns = len(sub)

# binary variable for regression
sex_male = np.zeros([ns],dtype=int)
sex_male[sex=='M'] = 1

# %% demographics summary plot

# bin_step = 5
# bins = np.arange(15, 75, step=bin_step)

bin_step = 10
bins = np.arange(20, 80, step=bin_step)
     
plt.figure()
plt.hist([age[sex=='M'],age[sex=='F']], bins, label=['M (N = '+str(sum(sex=='M'))+')','F (N = '+str(sum(sex=='F'))+')'],color=['orange','purple'], edgecolor='black')#,histtype='barstacked')
plt.legend(loc='upper right',prop={'size': lgs-4})
plt.ylim([0,5])
plt.xlabel('age (years)',size=lbs); plt.ylabel('# participants',size=lbs)
plt.xticks(size=axs); plt.yticks(size=axs);
plt.savefig(plot_dir+'/age_dist.png',bbox_inches='tight',dpi=600)

# %% days between scans

# loaded from secure .csv
scan_gap = [30,5,2,6,3,7,26,16,13,9,35,13,6,36,18,17,11,15,6,7,7,6,14]

bins = np.arange(0, 45, step=5)

plt.figure()
plt.hist(scan_gap, bins, color='lightgray', edgecolor='black')#,histtype='barstacked')
#plt.ylim([0,5])
plt.xlabel('time betw. 64mT scans (days)',size=lbs); plt.ylabel('# participants',size=lbs)
plt.xticks(size=axs); plt.yticks(size=axs);
plt.savefig(plot_dir+'/scan_gap_dist.png',bbox_inches='tight',dpi=600)
            
# %% project directories and session / acquisition lists

sseg_dir = [hype_dir+'/derivatives/synthseg_t1w', hype_dir+'/derivatives/synthseg_t2w']                         # segmentation volume .csv
#sreg_dir = [hype_dir+'/derivatives/synthseg_t1w-rigid-seg', hype_dir+'/derivatives/synthseg_t2w-rigid-seg']    # segmentation maps .nii RIGID
sreg_dir = [hype_dir+'/derivatives/synthseg_t1w-flirt-rigid', hype_dir+'/derivatives/synthseg_t2w-flirt-rigid'] # segmentation maps .nii RIGID
sreg_target = ['MPRAGE','T2w']                                                                                  # target high-field scans for registration

# corrected volume directories
sseg_correct_dir = [hype_dir+'/derivatives/synthseg_corrected_t1w', hype_dir+'/derivatives/synthseg_corrected_t2w']  

# sessions
ses = ['ses-HFC', 'ses-HFE']
nses = len(ses)

# contrasts
con = ['T1w', 'T2w']
ncon = len(con)

# acquisitions
acq = np.array([['acq-axi_T1w','acq-cor_T1w','acq-sag_T1w','acq-iso_T1w','T1w_mrr'],
                ['acq-axi_T2w','acq-cor_T2w','acq-sag_T2w','acq-iso_T2w','T2w_mrr']])
acq_plt = np.array([['T$_1$w AXI','T$_1$w COR','T$_1$w SAG','T$_1$w ISO','T$_1$w MRR'],
                ['T$_2$w AXI','T$_2$w COR','T$_2$w SAG','T$_2$w ISO','T$_2$w MRR']])
nacq = acq.shape[1]

# %% match .csv volumes and .nii segmentation labels

# SynthSeg *vol*umes: in .csv (101)
vol = np.array(pd.read_csv(sseg_dir[0]+'/sub-HYPE00/sub-HYPE00_ses-GE_acq-FSE_T1w_synthseg.csv').columns.values.tolist()[1:])
nvol = len(vol)

# SynthSeg ROI *seg*mentation IDs: in .nii (99)
seg_all = np.loadtxt('/Users/frantisek/Python/KCL/hyperfine/HYPE/synthseg_labels_all.txt',dtype='str')
seg_id = np.array([int(i[0]) for i in seg_all],dtype=int)
seg = np.array([i[1] for i in seg_all],dtype=str)
nseg = len(seg)

# # identical values to seg_id above
# seg_id = np.unique(np.array(nib.load(sreg_dir+'/sub-HYPE00/sub-HYPE00_ses-GE_acq-FSE_T1w_synthseg_to_MPRAGE.nii.gz').dataobj).flatten())

### manual IDs to match volumes and segmentation labels

# match volume IDs (from .csv)
vol_match = np.ones(nvol, bool)
vol_match[[0,2,20]] = False
vol[vol_match]

# match segmentation IDs (from .nii)
seg_match = np.ones(nseg, bool)
seg_match[[0]] = False
seg[seg_match]

### number of matching regions of interest (ROIs)
roi = seg[seg_match]
nroi = len(roi)

# %% global tissue volumes: GM-cortex / GM-subcortex / WM / CSF - MATCHED SEG REF (i.e. relative to 98/matched segmentation labels)

# names
gmc_nm = [elem for i, elem in enumerate(roi) if 'ctx' in elem]
gms_nm = ['Left-Thalamus-Proper','Left-Caudate','Left-Putamen','Left-Pallidum','Left-Hippocampus','Left-Amygdala','Left-Accumbens-area','Left-VentralDC',
          'Right-Thalamus-Proper','Right-Caudate','Right-Putamen','Right-Pallidum','Right-Hippocampus','Right-Amygdala','Right-Accumbens-area','Right-VentralDC']
wm_nm = ['Left-Cerebral-White-Matter','Right-Cerebral-White-Matter']
csf_nm = ['Left-Lateral-Ventricle','Left-Inf-Lat-Vent','3rd-Ventricle','4th-Ventricle','CSF','Right-Lateral-Ventricle','Right-Inf-Lat-Vent']

# use names to extract IDs (based on 98 ROIs)
gmc_ind = [i for i, elem in enumerate(roi) if 'ctx' in elem]                                        
gms_ind = [i for i, elem in enumerate(roi) if elem in gms_nm]     
wm_ind = [i for i, elem in enumerate(roi) if elem in wm_nm]                                                                          
csf_ind = [i for i, elem in enumerate(roi) if elem in csf_nm]

# exclusions
#roi[np.setdiff1d(np.arange(nroi), np.sort(np.concatenate((gmc_ind, gms_ind, wm_ind, csf_ind))))]  # exclusions (e.g. cerebellum: vol[6] / vol[24] + brainstem: vol[13])

# # check names / indices
# roi[gmc_ind]
# roi[gms_ind]
# roi[wm_ind]
# roi[csf_ind]

# global volumes
roi_g_nm = [gmc_nm, gms_nm, wm_nm, csf_nm]
roi_g_ind = [gmc_ind, gms_ind, wm_ind, csf_ind]
roi_g = ['GM-cort', 'GM-subcort', 'WM', 'CSF']
roi_g_plt = ['GM$_{cort}$', 'GM$_{subc}$', 'WM', 'CSF']
#roi_g_col_trt = ['mediumvioletred', 'purple', 'c', 'darkorange'] # ['mediumvioletred', 'purple', 'c', 'darkorange'] #['gold', 'orange', 'red', 'purple']
roi_g_col_trt = ['darkslategray','teal', 'darkturquoise', 'paleturquoise']
roi_g_col_lohi = ['saddlebrown','orangered','darkorange','orange']
nroi_g = len(roi_g)

# %% global tissue volumes: GM-cortex / GM-subcortex / WM / CSF - UNMATCHED VOL REF (i.e. relative to 101/unmatched volume labels)

# names
gmc_vol_nm = ['left cerebral cortex','right cerebral cortex']
gms_vol_nm = ['left thalamus', 'left caudate', 'left putamen', 'left pallidum','left hippocampus', 'left amygdala', 'left accumbens area','left ventral DC',
          'right thalamus', 'right caudate','right putamen', 'right pallidum', 'right hippocampus','right amygdala', 'right accumbens area', 'right ventral DC']
wm_vol_nm = ['left cerebral white matter', 'right cerebral white matter']
csf_vol_nm = ['left lateral ventricle', 'left inferior lateral ventricle','3rd ventricle', '4th ventricle', 'csf', 'right lateral ventricle','right inferior lateral ventricle']

# use names to extract IDs (based on 98 ROIs)
gmc_vol_ind = [i for i, elem in enumerate(vol) if elem in gmc_vol_nm]                                        
gms_vol_ind = [i for i, elem in enumerate(vol) if elem in gms_vol_nm]     
wm_vol_ind = [i for i, elem in enumerate(vol) if elem in wm_vol_nm]                                                                          
csf_vol_ind = [i for i, elem in enumerate(vol) if elem in csf_vol_nm]

# exclusions
#roi[np.setdiff1d(np.arange(nroi), np.sort(np.concatenate((gmc_ind, gms_ind, wm_ind, csf_ind))))]  # exclusions (e.g. cerebellum: vol[6] / vol[24] + brainstem: vol[13])

# check names / indices
vol[gmc_vol_ind]
vol[gms_vol_ind]
vol[wm_vol_ind]
vol[csf_vol_ind]

# global volumes
vol_g_nm = [gmc_vol_nm, gms_vol_nm, wm_vol_nm, csf_vol_nm]
vol_g_ind = [gmc_vol_ind, gms_vol_ind, wm_vol_ind, csf_vol_ind]
roi_g = ['GM-cort', 'GM-subcort', 'WM', 'CSF']
roi_g_plt = ['GM$_{cort}$', 'GM$_{subc}$', 'WM', 'CSF']
#roi_g_col_trt = ['mediumvioletred', 'purple', 'c', 'darkorange'] # ['mediumvioletred', 'purple', 'c', 'darkorange'] #['gold', 'orange', 'red', 'purple']
roi_g_col_trt = ['darkslategray','teal', 'darkturquoise', 'paleturquoise']
roi_g_col_lohi = ['saddlebrown','orangered','darkorange','orange']
nroi_g = len(roi_g)

# %%

"""
Scan visualisations (for one example participant)
"""

# %% anatomical scans for one example participant

if not os.path.isdir(plot_dir+'/example_nii'): os.mkdir(plot_dir+'/example_nii')

#import nilearn as nl
#from nilearn import plotting

# T1w scans
t1w_dir = hype_dir+'/derivatives/example-plots-sub-HYPE15'

# all T1w
for filename in os.listdir(t1w_dir):
    if filename.endswith(".nii.gz"):
        
        # file path and name
        file_path = os.path.join(t1w_dir, filename)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        print(file_name)
        
        # load
        img_nii = nib.load(file_path)
        img_vec = np.array(img_nii.dataobj).flatten()
        clim = (np.percentile(img_vec,1),np.percentile(img_vec,99))

        # plot
        plotting.plot_anat(img_nii,colorbar=False,display_mode='z',cut_coords=[35],annotate=False,vmin=clim[0],vmax=clim[1],output_file=plot_dir+'/example/'+file_name+'.png') # cmap=cmap_under, 


# T2w + FLAIR
for filename in os.listdir(t1w_dir+'/extra_reg_out'):
    if filename.endswith(".nii.gz"):
        
        # file path and name
        file_path = os.path.join(t1w_dir+'/extra_reg_out', filename)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        print(file_name)
        
        # load
        img_nii = nib.load(file_path)
        img_vec = np.array(img_nii.dataobj).flatten()
        clim = (np.percentile(img_vec,1),np.percentile(img_vec,99))

        # plot
        plotting.plot_anat(img_nii,colorbar=False,display_mode='z',cut_coords=[35],annotate=False,vmin=clim[0],vmax=clim[1],output_file=plot_dir+'/example_nii/'+file_name+'.png') # cmap=cmap_under, 

# %% SynthSeg illustration plot - all structures

# SynthSeg plot
#seg_all = np.loadtxt('/Users/frantisek/Python/KCL/hyperfine/HYPE/synthseg_labels_all.txt',dtype='str')
seg_col = np.array([[int(i[2]),int(i[3]),int(i[4])] for i in seg_all[:65]],dtype=int)/255
from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(seg_col, name='custom_cmap', N=len(seg_id[:65]))

seg_nii = nib.load(home_dir+'/Desktop/sub-HYPE15_ses-GE_acq-MPRAGE_T1w_synthseg.nii.gz')
seg_vec = np.array(seg_nii.dataobj).flatten()

seg_mask = (seg_vec!=0)*1 

# extract parameters from example volume
nvox = np.array(seg_nii.dataobj).size                                        # number of voxels
nii_shape = np.array(seg_nii.dataobj).shape                                  # image dimensions
nii_affine = seg_nii.affine         

seg_vec_lo = np.copy(seg_vec)
seg_vec_lo[seg_vec>=2000] = seg_vec[seg_vec>=2000]-1000

plot_vec = np.zeros(seg_vec.shape)
seg_col_lab = np.arange(1,66)
for i in range(65):
    plot_vec[seg_vec_lo==seg_id[:65][i]] = seg_col_lab[i]

# plot seg
f = plt.figure(figsize=(8, 4))
plot_nl_seg_masked(plot_vec, seg_mask, nii_shape, nii_affine, annotate=False, cmap=custom_cmap, display_mode='z', cut_coords=[35], figure=f, output_file=plot_dir+'/example_nii/sub-HYPE15_ses-HFC_T1w_mrr_synthseg.png')

# %% SynthSeg illustration plot - global classes

# SynthSeg plot
#seg_all = np.loadtxt('/Users/frantisek/Python/KCL/hyperfine/HYPE/synthseg_labels_all.txt',dtype='str')
seg_col = np.vstack(( np.zeros([1,3]) , np.array([colors.to_rgb(roi_g_col_trt[g])[:3] for g in range(nroi_g)]) ))
from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(seg_col, name='custom_cmap', N=len(seg_id[:65]))

seg_nii = nib.load(home_dir+'/Desktop/sub-HYPE15_ses-GE_acq-MPRAGE_T1w_synthseg.nii.gz')
seg_vec = np.array(seg_nii.dataobj).flatten()

seg_mask = (seg_vec!=0)*1 

# extract parameters from example volume
nvox = np.array(seg_nii.dataobj).size                                        # number of voxels
nii_shape = np.array(seg_nii.dataobj).shape                                  # image dimensions
nii_affine = seg_nii.affine         

#seg_vec_lo = np.copy(seg_vec)
#seg_vec_lo[seg_vec>=2000] = seg_vec[seg_vec>=2000]-1000

plot_vec = np.zeros(seg_vec.shape)
seg_col_lab = np.arange(1,nroi_g+1)
for g in range(nroi_g):
    for i in range(len(roi_g_nm[g])):
        plot_vec[seg_vec==seg_id[np.where(seg==roi_g_nm[g][i])[0][0]]] = seg_col_lab[g]
        #plot_vec[seg_vec_lo==seg_id[:65][i]] = seg_col_lab[i]

# plot seg
f = plt.figure(figsize=(8, 4))
plot_nl_seg_masked(plot_vec, seg_mask, nii_shape, nii_affine, annotate=False, cmap=custom_cmap, display_mode='z', cut_coords=[35], figure=f, output_file=plot_dir+'/example_nii/sub-HYPE15_ses-HFC_T1w_mrr_synthseg_glob_trt.png')

# %%

"""
Test-Retest Reliability i.e. "trt"
"""

# directory to save figures
if not os.path.isdir(plot_dir+'/trt_corr'): os.mkdir(plot_dir+'/trt_corr')
if not os.path.isdir(plot_dir+'/trt_dice'): os.mkdir(plot_dir+'/trt_dice')

# %% correct Hyperfine tissue volumes

for s in range(ns):
    print(f'--\nsubject: {sub[s]}\n--')
    for ss in range(nses):
        print(f'--\nsession: {ses[ss]}\n--')
        for c in range(ncon):
            print(f'--\ncontrast: {con[c]}\n--')
            for a in range(nacq):
                print(f'--\nacquisition: {acq[c,a]}\n--')
                # path to file
                csv_in_path = sseg_dir[c]+'/sub-'+sub[s]+'/sub-'+sub[s]+'_'+ses[ss]+'_'+acq[c,a]+'_synthseg.csv'
                csv_out_path = sseg_correct_dir[c]+'/sub-'+sub[s]+'/sub-'+sub[s]+'_'+ses[ss]+'_'+acq[c,a]+'_synthseg_corrected.csv'
                # if file exists, run correction function
                if (os.path.isfile(csv_in_path)):
                    if not os.path.isdir(sseg_correct_dir[c]+'/sub-'+sub[s]): os.mkdir(sseg_correct_dir[c]+'/sub-'+sub[s])
                    correct_synthseg_csv(csv_in_path,csv_out_path)
                else:
                    # QC - identify what data is missing (other than ses-HFE acq-iso_T2w, which is missing for all participants)
                    if ses[ss] != 'ses_HFE' and acq[c,a] != 'acq-iso_T2w':
                        print(f'missing: subject {sub[s]}, session {ses[ss]}, contrast {con[c]}, acquisition {acq[c,a]}')

# %% load *corrected* Hyperfine tissue volumes

### read t-rt volumes

# local tissue volumes
sseg_hf = np.zeros([ns,nses,ncon,nacq,nroi])
for s in range(ns):
    for ss in range(nses):
        for c in range(ncon):
            for a in range(nacq):
                # path to file
                csv_path = sseg_correct_dir[c]+'/sub-'+sub[s]+'/sub-'+sub[s]+'_'+ses[ss]+'_'+acq[c,a]+'_synthseg_corrected.csv'
                # if file exists, load it; else fill with NaN
                if (os.path.isfile(csv_path)):
                    sseg_hf[s,ss,c,a,:] = np.array(pd.read_csv(csv_path).iloc[0,1:],dtype=float)[vol_match]
                else:
                    sseg_hf[s,ss,c,a,:] = np.nan
                    # QC - identify what data is missing (other than ses-HFE acq-iso_T2w, which is missing for all participants)
                    if ses[ss] != 'ses_HFE' and acq[c,a] != 'acq-iso_T2w':
                        print(f'NaN: subject {sub[s]}, session {ses[ss]}, contrast {con[c]}, acquisition {acq[c,a]}')

# %% global tissue volumes

sseg_hf_g = np.zeros([ns,nses,ncon,nacq,nroi_g])
for s in range(ns):
    for ss in range(nses):
        for c in range(ncon):
            for a in range(nacq):
                # path to file
                csv_path = sseg_correct_dir[c]+'/sub-'+sub[s]+'/sub-'+sub[s]+'_'+ses[ss]+'_'+acq[c,a]+'_synthseg_corrected.csv'
                # if file exists, load it; else fill with NaN
                if (os.path.isfile(csv_path)):
                    # sum tissue volumes based on vol IDs
                    for g in range(nroi_g):
                        sseg_hf_g[s,ss,c,a,g] = np.sum(np.array(pd.read_csv(csv_path).iloc[0,1:],dtype=float)[vol_g_ind[g]])
                else:
                    sseg_hf_g[s,ss,c,a,:] = np.nan
                    # QC - identify what data is missing (other than ses-HFE acq-iso_T2w, which is missing for all participants)
                    if ses[ss] != 'ses_HFE' and acq[c,a] != 'acq-iso_T2w':
                        print(f'NaN: subject {sub[s]}, session {ses[ss]}, contrast {con[c]}, acquisition {acq[c,a]}')

# %% test-retest: calculate correlations

# local
print('----\nlocal\n----')
trt_icc = np.zeros([ncon,nacq,nroi])
trt_lin = np.zeros([ncon,nacq,nroi])
for c in range(ncon):
    print(f'--\ncontrast: {con[c]}\n--')
    for a in range(nacq):
        print(f'acquisition: {acq[c,a]}')
        for r in range(nroi):
            # IDs of subjects which don't contain NaN in either of the two sessions (the sum is taken over sessions)
            not_nan_id = np.setdiff1d(np.arange(ns),np.where(np.isnan(np.sum(sseg_hf[:,:,c,a,r],1)))[0])
            ## icc
            # if not_nan_id is not empty
            if len(not_nan_id) != 0:
                temp_icc = pg.intraclass_corr(data=pd.DataFrame({
                    'subject':np.concatenate((np.arange(0,len(not_nan_id)),np.arange(0,len(not_nan_id)))),
                    'scan':np.concatenate((np.repeat('T',len(not_nan_id)),np.repeat('RT',len(not_nan_id)))),
                    'score':np.concatenate((sseg_hf[not_nan_id,0,c,a,r],sseg_hf[not_nan_id,1,c,a,r]))}),
                    targets='subject', raters='scan',ratings='score')
                # statistics
                trt_icc[c,a,r] = temp_icc['ICC'][2] 
                trt_lin[c,a,r] = lin(sseg_hf[not_nan_id,0,c,a,r],sseg_hf[not_nan_id,1,c,a,r])
            # else if not_nan_id is empty (i.e. all values are NaN)
            else:
                trt_icc[c,a,r] = np.nan 
                trt_lin[c,a,r] = np.nan

# global
print('----\nglobal\n----')
trt_icc_g = np.zeros([ncon,nacq,nroi_g])
trt_lin_g = np.zeros([ncon,nacq,nroi_g])
for c in range(ncon):
    print(f'--\ncontrast: {con[c]}\n--')
    for a in range(nacq):
        print(f'acquisition: {acq[c,a]}')
        for g in range(nroi_g):
            # IDs of subjects which don't contain NaN in either of the two sessions (the sum is taken over sessions)
            not_nan_id = np.setdiff1d(np.arange(ns),np.where(np.isnan(np.sum(sseg_hf_g[:,:,c,a,g],1)))[0])
            ## icc
            # if not_nan_id is not empty
            if len(not_nan_id) != 0:
                temp_icc = pg.intraclass_corr(data=pd.DataFrame({
                    'subject':np.concatenate((np.arange(0,len(not_nan_id)),np.arange(0,len(not_nan_id)))),
                    'scan':np.concatenate((np.repeat('T',len(not_nan_id)),np.repeat('RT',len(not_nan_id)))),
                    'score':np.concatenate((sseg_hf_g[not_nan_id,0,c,a,g],sseg_hf_g[not_nan_id,1,c,a,g]))}),
                    targets='subject', raters='scan',ratings='score')
                # statistics
                trt_icc_g[c,a,g] = temp_icc['ICC'][2] 
                trt_lin_g[c,a,g] = lin(sseg_hf_g[not_nan_id,0,c,a,g],sseg_hf_g[not_nan_id,1,c,a,g])
            # else if not_nan_id is empty (i.e. all values are NaN)
            else:
                trt_icc_g[c,a,g] = np.nan 
                trt_lin_g[c,a,g] = np.nan

# %% test-retest global tissue volume plots

for c in range(ncon):
    for a in range(nacq):
    #for a in range(len(acq_subset_id)):

        # IDs of subjects which don't contain NaN in either of the two sessions (the sum is taken over sessions)
        not_nan_id = np.setdiff1d(np.arange(ns),np.where(np.isnan(np.sum(sseg_hf_g[:,:,c,a,0],1)))[0]) # g = 0
        #not_nan_id = np.setdiff1d(np.arange(ns),np.where(np.isnan(np.sum(sseg_hf_g[:,:,c,acq_subset_id[a],0],1)))[0]) # g = 0
        
        if len(not_nan_id) != 0:
        
            fig = plt.figure(figsize=(13, 4.5))#, layout='compressed') # layout='constrained')#
        
            for g in range(nroi_g):
                
                plt.subplot(1,nroi_g,g+1, aspect='equal')
    
                # x = np.copy(sseg_hf_g[not_nan_id,0,c,acq_subset_id[a],g]/1e3)
                # y = np.copy(sseg_hf_g[not_nan_id,1,c,acq_subset_id[a],g]/1e3)
                x = np.copy(sseg_hf_g[not_nan_id,0,c,a,g]/1e3)
                y = np.copy(sseg_hf_g[not_nan_id,1,c,a,g]/1e3)
                
                # limits and ticks
                ax_min = 0.9*np.min([np.min(x),np.min(y)])
                ax_max = 1.07*np.max([np.max(x),np.max(y)])
                ax_lim = [ax_min, ax_max]
                
                # plot
                plt.plot([ax_lim[0]-1,ax_lim[1]+1], [ax_lim[0]-1,ax_lim[1]+1], ls='--', c='gray')   # plot dashed identity line
                sb.regplot(x, y, ci=95, scatter_kws={"color": roi_g_col_trt[g]}, line_kws={"color": "black"}) # line of best fit + CI
                plt.xticks(fontsize=axs-2); plt.xlim(ax_lim);  # plt.xticks(fontsize=axs-1);        # axis limits and ticks
                plt.yticks(plt.gca().get_xticks(),fontsize=axs-2); plt.ylim(ax_lim);  # plt.yticks(fontsize=axs-1); 
    
                # title and save
                #plt.title(roi_g_plt[g]+'; ICC = '+str(round(trt_icc_g[c,acq_subset_id[a],g],2)),fontsize=lbs-2,pad=13)
                #plt.title(roi_g_plt[g]+'; ICC = '+str(round(trt_icc_g[c,a,g],2)),fontsize=lbs-2,pad=13) # with ROI label
                plt.title('ICC = '+str(round(trt_icc_g[c,a,g],2)),fontsize=lbs-1,pad=13)

            fig.tight_layout(rect=[0.05, 0.01, 1, 0.95])
            #fig.suptitle(acq_plt[c,acq_subset_id[a]], fontsize=lbs+1)
            #fig.suptitle(acq_plt[c,a], fontsize=lbs+1)                 # with acq title
            fig.supxlabel('64mT test volume (cm$^3$)',fontsize=lbs)
            fig.supylabel('64mT retest vol. (cm$^3$)',fontsize=lbs)
            
            #if save_fig: plt.savefig(plot_dir+'/trt_corr/trt_'+acq[c,acq_subset_id[a]]+'.png',dpi=600) # ,bbox_inches='tight'
            if save_fig: 
                plt.savefig(plot_dir+'/trt_corr/trt_'+acq[c,a]+'.png',dpi=600) # ,bbox_inches='tight'
                plt.savefig(plot_dir+'/trt_corr/trt_'+acq[c,a]+'.svg')
            
            else:
                continue

# %% test-retest results local tissue volume plots (whole-brain "segmentation" plots)

# ICC (T1w / T2w)           trt_icc                     (2, 5, 98)
# mean Dice (T1w / T2w)     np.nanmedian(trt_dice,0)    (2, 5, 98)

# MNI template volume
seg_nii = nib.load(hype_dir+'/mni/mni_synthseg.nii')
seg_vec = np.array(seg_nii.dataobj).flatten()
seg_mask = (seg_vec!=0)*1

# extract parameters from example volume
nvox = np.array(seg_nii.dataobj).size           # number of voxels
nii_shape = np.array(seg_nii.dataobj).shape     # image dimensions
nii_affine = seg_nii.affine                     # affine

for c in range(ncon):
    for a in range(nacq):
        
        # skip T2w iso (for ICC)
        if c==1 and a==3: continue
        
        ### ICC
        
        # fill vector
        plot_vec = np.zeros(seg_vec.shape)
        for r in range(nroi):
            plot_vec[seg_vec==seg_id[seg_match][r]] = trt_icc[c,a,r]
        
        # plot seg
        f = plt.figure(figsize=(8, 4))
        cl = [deci_floor(np.min(trt_icc[c,a,:]),1),deci_ceil(np.max(trt_icc[c,a,:]),1)]
        plot_nl_image_masked(plot_vec, seg_mask, nii_shape, nii_affine, annotate=False, cmap='viridis_r', clim=cl, cut_coords=np.array([-20,0,20]), draw_cross=True,black_bg=False,display_mode='z',figure=f,output_file=plot_dir+'/trt_corr/trt_mni_icc_'+acq[c,a]+'.png')

 # %% write global and local ICC results

nround = 3                                   # number of decimal places to round
filename = plot_dir+'/trt_corr/trt_icc.csv'  # output

with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)

    for c in range(ncon):
    
        # header
        writer.writerow(['statistic'] + list(acq[c,:]))
        
        # global median
        writer.writerow(np.concatenate((np.array(['ICC glob Md']),np.round(np.nanmedian(trt_icc_g, 2)[c,:],nround))))
        
        # local median [q1,q3]
        qrow = ['ICC loc Md [Q1, Q3]']
        for i in range(nacq):
            qrow += [f'{round(np.nanmedian(trt_icc, 2)[c,i],nround)} [{round(np.nanpercentile(trt_icc, 25, 2)[c,i],nround)},{round(np.nanpercentile(trt_icc, 75, 2)[c,i],nround)}]']
        writer.writerow(qrow)
        
        # blank row (between contrasts)
        writer.writerow(' ')

# %% bias - relationship of ICC to average (high-field) volumes

# *** check use of nanmean ***
# *** check shape of sseg_ge ***

trt_icc_bias_rho, trt_icc_bias_p = np.zeros([ncon,nacq]), np.zeros([ncon,nacq])
for c in range(ncon):
    for a in range(nacq):
        trt_icc_bias_rho[c,a] = stats.spearmanr(np.nanmedian(sseg_ge[:,c,:],0),trt_icc[c,a,:])[0]
        trt_icc_bias_p[c,a] = stats.spearmanr(np.nanmedian(sseg_ge[:,c,:],0),trt_icc[c,a,:])[1]

# %% write ICC bias results

nround = 3                                      # number of decimal places to round
filename = plot_dir+'/trt_corr/trt_icc_bias.csv'  # output

with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)

    for c in range(ncon):
    
        # header
        writer.writerow(['statistic'] + list(acq[c,:]))
        
        # rho & p
        writer.writerow(np.concatenate((np.array(['ICC Spearman\'s rho']),np.round(trt_icc_bias_rho[c,:],nround))))
        writer.writerow(np.concatenate((np.array(['ICC Spearman\'s p']),np.array(['{:g}'.format(float('{:.{p}g}'.format(i, p=nround))) for i in trt_icc_bias_p[c,:]]))))
        
        # blank row (between contrasts)
        writer.writerow(' ')

# %% test-retest: segmentations - RIGID (takes a long time to run)

trt_dice_g = np.zeros([ns,ncon,nacq,nroi_g])
trt_dice = np.zeros([ns,ncon,nacq,nroi])
for s in range(ns):
    print(f'----\nsubject {str(s)}\n----')
    for c in range(ncon):
        print(f'--\ncontrast: {con[c]}\n--')
        for a in range(nacq):
            print(f'acquisition: {acq[c,a]}')
            # paths to files
            #hfc_nii_path = sreg_dir[c]+'/sub-'+sub[s]+'/sub-'+sub[s]+'_ses-HFC_'+acq[c,a]+'_synthseg_to_'+sreg_target[c]+'.nii.gz'
            #hfe_nii_path = sreg_dir[c]+'/sub-'+sub[s]+'/sub-'+sub[s]+'_ses-HFE_'+acq[c,a]+'_synthseg_to_'+sreg_target[c]+'.nii.gz'
            hfc_nii_path = sreg_dir[c]+'/sub-'+sub[s]+'/sub-'+sub[s]+'_ses-HFC_'+acq[c,a]+'_synthseg_flirt_to_ses-GE_'+sreg_target[c]+'.nii.gz'
            hfe_nii_path = sreg_dir[c]+'/sub-'+sub[s]+'/sub-'+sub[s]+'_ses-HFE_'+acq[c,a]+'_synthseg_flirt_to_ses-GE_'+sreg_target[c]+'.nii.gz'
            
            # if files exist, load them...
            if (os.path.isfile(hfc_nii_path)) and (os.path.isfile(hfe_nii_path)):
                
                # load files
                hfc = np.array(nib.load(hfc_nii_path).dataobj).flatten()
                hfe = np.array(nib.load(hfe_nii_path).dataobj).flatten()
                
                # dice global 
                for g in range(nroi_g):
                    hfc_g,hfe_g = np.zeros(len(hfc)),np.zeros(len(hfe)) # initialise empty variable
                    for gr in range(len(roi_g_ind[g])):                 # loop over aggregate indices for current tissue class
                        hfc_g[hfc==seg_id[seg_match][roi_g_ind[g][gr]]] = 1                # fill temporary vector for current class - hfc
                        hfe_g[hfe==seg_id[seg_match][roi_g_ind[g][gr]]] = 1                # fill temporary vector for current class - hfc
                    trt_dice_g[s,c,a,g] = 1-dice(hfc_g!=0,hfe_g!=0)     # calculate dice
                
                # # alternative one-line approach to dice global, but takes ~19-20s per scan
                # for g in range(nroi_g):
                #     trt_dice_g[s,c,a,g] = 1-dice(np.any(np.repeat(np.expand_dims(hfc, 1),len(roi_g_ind[g]),axis=1)==roi_g_ind[g],1), np.any(np.repeat(np.expand_dims(hfe, 1),len(roi_g_ind[g]),axis=1)==roi_g_ind[g],1))
               
                # dice local
                for r in range(nroi):
                    trt_dice[s,c,a,r] = 1-dice(hfc==seg_id[seg_match][r],hfe==seg_id[seg_match][r])
               
            # ...else fill with NaNs
            else:
                trt_dice_g[s,c,a,:] = np.nan
                trt_dice[s,c,a,:] = np.nan

np.save(plot_dir+'/trt_dice/trt_dice_g.npy', trt_dice_g)
np.save(plot_dir+'/trt_dice/trt_dice.npy', trt_dice)

# %% load pre-saved dice results

trt_dice_g = np.load(plot_dir+'/trt_dice/trt_dice_g.npy')
trt_dice = np.load(plot_dir+'/trt_dice/trt_dice.npy')

# %% test-retest global Dice plots

import ptitprince as pt          # raincloud plots
['gold', 'orange', 'red']

for c in range(ncon):
    for a in range(nacq):
    
        # IDs of subjects which don't contain NaN in either of the two sessions (the sum is taken over sessions)
        not_nan_id = np.setdiff1d(np.arange(ns),np.where(np.isnan(trt_dice_g[:,c,a,0]))[0]) # g = 0
    
        if len(not_nan_id) != 0:
    
            #fig = plt.figure(figsize=(8, 5))#, layout='compressed') # layout='constrained')
            f, ax = plt.subplots(figsize=(5, 5))
            
            dx = list( np.concatenate(( np.repeat(0,len(not_nan_id)), np.repeat(1,len(not_nan_id)), np.repeat(2,len(not_nan_id)), np.repeat(3,len(not_nan_id)) )) )
            dy = list( trt_dice_g[not_nan_id,c,a,:].T.flatten() )

            pt.RainCloud(x = dx, y = dy, palette = roi_g_col_trt, bw = .4, width_viol = 0.5, orient = "v", box_showfliers=False)
            ax.set_xticklabels(roi_g_plt, size=lbs-2); #plt.xticks(fontsize=lbs+2); 
            plt.ylabel('Dice coeff.', size=lbs+4); plt.yticks(fontsize=lbs-5); 
            plt.xlim([-0.65,3.35])
            plt.ylim([0.2,1.0])
            plt.title(acq_plt[c,a], fontsize=lbs+1)
            #f.tight_layout()
            #fig.tight_layout(rect=[0.05, 0.01, 1, 0.95])
            
            #if save_fig: plt.savefig(plot_dir+'/trt_corr/trt_'+acq[c,acq_subset_id[a]]+'.png',dpi=600) # ,bbox_inches='tight'
            #if save_fig: plt.savefig(plot_dir+'/trt_dice/trt_dice_'+acq[c,a]+'.png',dpi=600) # ,bbox_inches='tight'
            if save_fig: plt.savefig(plot_dir+'/trt_dice/trt_dice_'+acq[c,a]+'.svg') # ,bbox_inches='tight'
        
            else:
                continue

# %% test-retest local Dice plots (whole-brain "segmentation" plots)

# ICC (T1w / T2w)           trt_icc                     (2, 5, 98)
# mean Dice (T1w / T2w)     np.nanmedian(trt_dice,0)    (2, 5, 98)

# MNI template volume
seg_nii = nib.load(hype_dir+'/mni/mni_synthseg.nii')
seg_vec = np.array(seg_nii.dataobj).flatten()
seg_mask = (seg_vec!=0)*1

# extract parameters from example volume
nvox = np.array(seg_nii.dataobj).size           # number of voxels
nii_shape = np.array(seg_nii.dataobj).shape     # image dimensions
nii_affine = seg_nii.affine                     # affine

for c in range(ncon):
    for a in range(nacq):
        
        # skip T2w iso (for t-rt)
        if c==1 and a==3: continue
        
        ### Dice
        
        # fill vector
        plot_vec = np.zeros(seg_vec.shape)
        for r in range(nroi):
            # if sum(seg_vec==seg_id[seg_match][i])==0:
            #     print(f'contrast {c}, acquisition {a}, ROI {i} empty')
            plot_vec[seg_vec==seg_id[seg_match][r]] = np.nanmedian(trt_dice,0)[c,a,r]
        
        # plot seg
        f = plt.figure(figsize=(8, 4))
        cl = [deci_floor(np.min(np.nanmedian(trt_dice,0)[c,a,:]),1),deci_ceil(np.max(np.nanmedian(trt_dice,0)[c,a,:]),1)]
        plot_nl_image_masked(plot_vec, seg_mask, nii_shape, nii_affine, annotate=False, cmap='viridis_r', clim=cl, cut_coords=np.array([-20,0,20]), draw_cross=True,black_bg=False,display_mode='z',figure=f,output_file=plot_dir+'/trt_dice/trt_mni_md_dice_'+acq[c,a]+'.png')

# %% write Dice results

nround = 3                                      # number of decimal places to round
filename = plot_dir+'/trt_dice/trt_dice.csv'  # output

with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)

    for c in range(ncon):
    
        # header
        writer.writerow(['statistic'] + list(acq[c,:]))
        
        # global median
        writer.writerow(np.concatenate((np.array(['Md Dice glob Md']),np.round(np.nanmedian(np.nanmedian(trt_dice_g,0),2)[c,:],nround))))
        
        # local median [q1,q3]
        qrow = ['Md Dice loc Md [Q1, Q3]']
        for i in range(nacq):
            qrow += [f'{round(np.nanmedian(np.nanmedian(trt_dice,0), 2)[c,i],nround)} [{round(np.nanpercentile(np.nanmedian(trt_dice,0), 25, 2)[c,i],nround)},{round(np.nanpercentile(np.nanmedian(trt_dice,0), 75, 2)[c,i],nround)}]']
        writer.writerow(qrow)
        
        # blank row (between contrasts)
        writer.writerow(' ')

# %% Dice bias?

trt_dice_bias_rho, trt_dice_bias_p = np.zeros([ncon,nacq]), np.zeros([ncon,nacq])
for c in range(ncon):
    for a in range(nacq):
        trt_dice_bias_rho[c,a] = stats.spearmanr(np.nanmedian(sseg_ge[:,c,:],0),np.nanmedian(trt_dice,0)[c,a,:])[0]
        trt_dice_bias_p[c,a] = stats.spearmanr(np.nanmedian(sseg_ge[:,c,:],0),np.nanmedian(trt_dice,0)[c,a,:])[1]

# %% write Dice bias results

nround = 3                                      # number of decimal places to round
filename = plot_dir+'/trt_dice/trt_dice_bias.csv'  # output

with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)

    for c in range(ncon):
    
        # header
        writer.writerow(['statistic'] + list(acq[c,:]))
        
        # rho & p
        writer.writerow(np.concatenate((np.array(['Md Dice Spearman\'s rho']),np.round(trt_dice_bias_rho[c,:],nround))))        
        writer.writerow(np.concatenate((np.array(['Md Dice Spearman\'s p']),np.array(['{:g}'.format(float('{:.{p}g}'.format(i, p=nround))) for i in trt_dice_bias_p[c,:]]))))
        
        # blank row (between contrasts)
        writer.writerow(' ')
        
# %%

"""
Ultra-Low-Field VS High-Field, i.e. "lohi"
"""

# directory to save figures
if not os.path.isdir(plot_dir+'/lohi_corr'): os.mkdir(plot_dir+'/lohi_corr')
if not os.path.isdir(plot_dir+'/lohi_diff'): os.mkdir(plot_dir+'/lohi_diff')
if not os.path.isdir(plot_dir+'/lohi_dice'): os.mkdir(plot_dir+'/lohi_dice')

# %% correct high-field tissue volumes

sseg_ge_str = ['acq-MPRAGE_T1w','T2w']

for s in range(ns):
    print(f'--\nsubject: {sub[s]}\n--')
    for c in range(ncon):
        print(f'--\ncontrast: {con[c]}\n--')
        # path to file
        ref_csv_in_path = sseg_dir[c]+'/sub-'+sub[s]+'/sub-'+sub[s]+'_ses-GE_'+sseg_ge_str[c]+'_synthseg.csv'
        ref_csv_out_path = sseg_correct_dir[c]+'/sub-'+sub[s]+'/sub-'+sub[s]+'_ses-GE_'+sseg_ge_str[c]+'_synthseg_corrected.csv'
        # if file exists, run correction function
        if (os.path.isfile(ref_csv_in_path)):
            if not os.path.isdir(sseg_correct_dir[c]+'/sub-'+sub[s]): os.mkdir(sseg_correct_dir[c]+'/sub-'+sub[s])
            correct_synthseg_csv(ref_csv_in_path,ref_csv_out_path)
        else:
            # QC - identify what data is missing
            print(f'NaN: subject {sub[s]}, contrast {con[c]}')

# %% load *corrected* high-field (GE) volumes

sseg_ge_str = ['acq-MPRAGE_T1w','T2w']

# read high-field scan local volumes
sseg_ge = np.zeros([ns,ncon,nroi])
for s in range(ns):
    for c in range(ncon):
        # path to file
        ref_csv_path = sseg_correct_dir[c]+'/sub-'+sub[s]+'/sub-'+sub[s]+'_ses-GE_'+sseg_ge_str[c]+'_synthseg_corrected.csv'
        # if file exists, load it; else fill with NaN
        if (os.path.isfile(ref_csv_path)):
            sseg_ge[s,c,:] = np.array(pd.read_csv(ref_csv_path).iloc[0,1:],dtype=float)[vol_match]
        else:
            sseg_ge[s,c,:] = np.nan
            # QC - identify what data is missing
            print(f'NaN: subject {sub[s]}, contrast {con[c]}')

# %% high-field (GE) volumes - cortical aggregate volume

# aggregate high-field scan global volumes
sseg_ge_g = np.zeros([ns,ncon,nroi_g])
for s in range(ns):
    for c in range(ncon):
        # path to file
        ref_csv_path = sseg_correct_dir[c]+'/sub-'+sub[s]+'/sub-'+sub[s]+'_ses-GE_'+sseg_ge_str[c]+'_synthseg_corrected.csv'
        # if file exists, load it; else fill with NaN
        if (os.path.isfile(ref_csv_path)):
            # sum tissue volumes based on vol IDs
            for g in range(nroi_g):
                sseg_ge_g[s,c,g] = np.sum(np.array(pd.read_csv(ref_csv_path).iloc[0,1:],dtype=float)[vol_g_ind[g]])
        else:
            sseg_ge_g[s,c,:] = np.nan
            # QC - identify what data is missing
            print(f'NaN: subject {sub[s]}, contrast {con[c]}')

# %% calculate correlations between ULF and HF

# reminder of sseg_hf shape:
# sseg_hf = np.zeros([ns,nses,ncon,nacq,nroi])

# local
lohi_r = np.zeros([ncon,nacq,nroi])
lohi_lin = np.zeros([ncon,nacq,nroi])
for c in range(ncon):
    print(f'--\ncontrast: {con[c]}\n--')
    for a in range(nacq):
        print(f'acquisition: {acq[c,a]}')
        for r in range(nroi):
            # IDs of subjects which don't contain NaN (in the HFC / first = 0 session only, i.e. sseg_hf[:,0,c,a,r])
            not_nan_id = np.setdiff1d(np.arange(ns),np.where(np.isnan(sseg_hf[:,0,c,a,r]))[0])
            # statistics
            lohi_r[c,a,r] = stats.pearsonr(sseg_hf[:,0,c,a,r][not_nan_id],sseg_ge[:,c,r][not_nan_id])[0]
            lohi_lin[c,a,r] = lin(sseg_hf[:,0,c,a,r][not_nan_id],sseg_ge[:,c,r][not_nan_id])

# global
lohi_r_g = np.zeros([ncon,nacq,nroi_g])
lohi_lin_g = np.zeros([ncon,nacq,nroi_g])
for c in range(ncon):
    print(f'--\ncontrast: {con[c]}\n--')
    for a in range(nacq):
        print(f'acquisition: {acq[c,a]}')
        for g in range(nroi_g):
            # IDs of subjects which don't contain NaN
            not_nan_id = np.setdiff1d(np.arange(ns),np.where(np.isnan(sseg_hf_g[:,0,c,a,g]))[0])
            # statistics
            lohi_r_g[c,a,g] = stats.pearsonr(sseg_hf_g[:,0,c,a,g][not_nan_id],sseg_ge_g[:,c,g][not_nan_id])[0]
            lohi_lin_g[c,a,g] = lin(sseg_hf_g[:,0,c,a,g][not_nan_id],sseg_ge_g[:,c,g][not_nan_id])

# %% ulf-hf global tissue volume plots

for c in range(ncon):
    for a in range(nacq):

        # IDs of subjects which don't contain NaN in the HFC session
        not_nan_id = np.setdiff1d(np.arange(ns),np.where(np.isnan(sseg_hf_g[:,0,c,a,0]))[0]) # ss = 0, g = 0

        if len(not_nan_id) != 0:
        
            fig = plt.figure(figsize=(13, 4.5))
        
            for g in range(nroi_g):
                
                plt.subplot(1,nroi_g,g+1, aspect='equal')
    
                x = np.copy(sseg_hf_g[not_nan_id,0,c,a,g]/1e3)
                y = np.copy(sseg_ge_g[not_nan_id,c,g]/1e3)
                
                # limits and ticks
                ax_min = 0.9*np.min([np.min(x),np.min(y)])
                ax_max = 1.07*np.max([np.max(x),np.max(y)])
                ax_lim = [ax_min, ax_max]
                
                # plot
                plt.plot([ax_lim[0]-1,ax_lim[1]+1], [ax_lim[0]-1,ax_lim[1]+1], ls='--', c='gray')   # plot dashed identity line
                sb.regplot(x, y, ci=95, scatter_kws={"color": roi_g_col_lohi[g]}, line_kws={"color": "black"}) # line of best fit + CI
                plt.xticks(fontsize=axs-2); plt.xlim(ax_lim);  # plt.xticks(fontsize=axs-1);        # axis limits and ticks
                plt.yticks(plt.gca().get_xticks(),fontsize=axs-2); plt.ylim(ax_lim);  # plt.yticks(fontsize=axs-1); 
    
                # title and save
                #plt.title(roi_g_plt[g]+'\n r = '+str(round(lohi_r_g[c,a,g],2))+'; CCC = '+str(round(lohi_lin_g[c,a,g],2)),fontsize=lbs-2,pad=13, linespacing=1.65)
                plt.title('r = '+str(round(lohi_r_g[c,a,g],2))+'; CCC = '+str(round(lohi_lin_g[c,a,g],2)),fontsize=lbs-1,pad=13, linespacing=1.65)

            fig.tight_layout(rect=[0.05, 0.07, 1, 0.88])
            #fig.suptitle(acq_plt[c,a], fontsize=lbs+1)
            fig.supxlabel('64mT volume (cm$^3$)',fontsize=lbs)
            fig.supylabel('3T volume (cm$^3$)',fontsize=lbs)
            
            #if save_fig: plt.savefig(plot_dir+'/lohi_corr/lohi_'+acq[c,acq_subset_id[a]]+'.png',dpi=600) # ,bbox_inches='tight'
            if save_fig: 
                plt.savefig(plot_dir+'/lohi_corr/lohi_'+acq[c,a]+'.png',dpi=600) # ,bbox_inches='tight'
                plt.savefig(plot_dir+'/lohi_corr/lohi_'+acq[c,a]+'.svg') 
            
            else:
                continue

# %% local tissue volume plots (whole-brain "segmentation" plots)

# Pearson's r (T1w / T2w)   lohi_r                      (2, 5, 98)
# Lin's CCC (T1w / T2w)     lohi_lin                    (2, 5, 98)

# MNI template volume
seg_nii = nib.load(hype_dir+'/mni/mni_synthseg.nii')
seg_vec = np.array(seg_nii.dataobj).flatten()
seg_mask = (seg_vec!=0)*1

# extract parameters from example volume
nvox = np.array(seg_nii.dataobj).size           # number of voxels
nii_shape = np.array(seg_nii.dataobj).shape     # image dimensions
nii_affine = seg_nii.affine                     # affine

for c in range(ncon):
    for a in range(nacq):
        
        ### Pearson's r
        
        # fill vector
        plot_vec = np.zeros(seg_vec.shape)
        for r in range(nroi):
            plot_vec[seg_vec==seg_id[seg_match][r]] = lohi_r[c,a,r]
        
        # plot seg
        f = plt.figure(figsize=(8, 4))
        cl = [0,1]
        plot_nl_image_masked(plot_vec, seg_mask, nii_shape, nii_affine, annotate=False, cmap='plasma_r', clim=cl, cut_coords=np.array([-20,0,20]), draw_cross=True,black_bg=False,display_mode='z',figure=f,output_file=plot_dir+'/lohi_corr/lohi_mni_pearson_'+acq[c,a]+'.png')

        ### Lin's ICC
        
        # fill vector
        plot_vec = np.zeros(seg_vec.shape)
        for r in range(nroi):
            plot_vec[seg_vec==seg_id[seg_match][r]] = lohi_lin[c,a,r]
        
        # plot seg
        f = plt.figure(figsize=(8, 4))
        cl = [0,1]
        plot_nl_image_masked(plot_vec, seg_mask, nii_shape, nii_affine, annotate=False, cmap='plasma_r', clim=cl, cut_coords=np.array([-20,0,20]), draw_cross=True,black_bg=False,display_mode='z',figure=f,output_file=plot_dir+'/lohi_corr/lohi_mni_lin_'+acq[c,a]+'.png')

# %% write lo-hi results

nround = 3                                          # number of decimal places to round
filename = plot_dir+'/lohi_corr/lohi_corr.csv'      # output

with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)

    for c in range(ncon):
    
        # header
        writer.writerow(['statistic'] + list(acq[c,:]))
        
        ### Pearson's r
        
        # global median
        writer.writerow(np.concatenate((np.array(['Pearson\'s r glob Md']),np.round(np.nanmedian(lohi_r_g, 2)[c,:],nround))))
        
        # local median [q1,q3]
        qrow = ['Pearson\'s r loc Md [Q1, Q3]']
        for a in range(nacq):
            qrow += [f'{round(np.nanmedian(lohi_r, 2)[c,a],nround)} [{round(np.nanpercentile(lohi_r, 25, 2)[c,a],nround)},{round(np.nanpercentile(lohi_r, 75, 2)[c,a],nround)}]']
        writer.writerow(qrow)
        
        ### Lin's CCC
        
        # global median
        writer.writerow(np.concatenate((np.array(['Lin\'s CCC glob Md']),np.round(np.nanmedian(lohi_lin_g, 2)[c,:],nround))))
        
        # local median [q1,q3]
        qrow = ['Lin\'s CCC loc Md [Q1, Q3]']
        for a in range(nacq):
            qrow += [f'{round(np.nanmedian(lohi_lin, 2)[c,a],nround)} [{round(np.nanpercentile(lohi_lin, 25, 2)[c,a],nround)},{round(np.nanpercentile(lohi_lin, 75, 2)[c,a],nround)}]']
        writer.writerow(qrow)
        
        # blank row (between contrasts)
        writer.writerow(' ')

# %% bias - relationship of r / Lin to Median (high-field) volumes

lohi_r_bias_rho, lohi_r_bias_p = np.zeros([ncon,nacq]), np.zeros([ncon,nacq])
lohi_lin_bias_rho, lohi_lin_bias_p = np.zeros([ncon,nacq]), np.zeros([ncon,nacq])
for c in range(ncon):
    for a in range(nacq):
        # Pearson's r
        lohi_r_bias_rho[c,a] = stats.spearmanr(np.nanmedian(sseg_ge[:,c,:],0),lohi_r[c,a,:])[0]
        lohi_r_bias_p[c,a] = stats.spearmanr(np.nanmedian(sseg_ge[:,c,:],0),lohi_r[c,a,:])[1]
        # Lin's CCC
        lohi_lin_bias_rho[c,a] = stats.spearmanr(np.nanmedian(sseg_ge[:,c,:],0),lohi_lin[c,a,:])[0]
        lohi_lin_bias_p[c,a] = stats.spearmanr(np.nanmedian(sseg_ge[:,c,:],0),lohi_lin[c,a,:])[1]

# %% write lo-hi bias results

nround = 3                                          # number of decimal places to round
filename = plot_dir+'/lohi_corr/lohi_corr_bias.csv'   # output

with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)

    for c in range(ncon):
    
        # header
        writer.writerow(['statistic'] + list(acq[c,:]))
        
        ### Pearson's r
        
        # rho & p
        writer.writerow(np.concatenate((np.array(['Pear. r Spearman\'s rho']),np.round(lohi_r_bias_rho[c,:],nround))))
        writer.writerow(np.concatenate((np.array(['Pear. r Spearman\'s p']),np.array(['{:g}'.format(float('{:.{p}g}'.format(i, p=nround))) for i in lohi_r_bias_p[c,:]]))))
        
        ### Lin's CCC
        writer.writerow(np.concatenate((np.array(['Lin\'s CCC Spearman\'s rho']),np.round(lohi_lin_bias_rho[c,:],nround))))
        writer.writerow(np.concatenate((np.array(['Lin\'s CCC Spearman\'s p']),np.array(['{:g}'.format(float('{:.{p}g}'.format(i, p=nround))) for i in lohi_lin_bias_p[c,:]]))))
        
        # blank row (between contrasts)
        writer.writerow(' ')

# %% under- or over-estimate?

# local
lohi_wsr_rb, lohi_wsr_p = np.zeros([ncon,nacq,nroi]), np.zeros([ncon,nacq,nroi])
for c in range(ncon):
    for a in range(nacq):
        for r in range(nroi):
            # IDs of subjects which don't contain NaN
            not_nan_id = np.setdiff1d(np.arange(ns),np.where(np.isnan(sseg_hf[:,0,c,a,r]))[0])
            # statistics
            lohi_wsr_rb[c,a,r] = pg.wilcoxon(sseg_hf[:,0,c,a,r][not_nan_id],sseg_ge[:,c,r][not_nan_id])['RBC']
            lohi_wsr_p[c,a,r] = pg.wilcoxon(sseg_hf[:,0,c,a,r][not_nan_id],sseg_ge[:,c,r][not_nan_id])['p-val']

# proportions per scan
np.sum(lohi_wsr_rb>0,2) # overestimate (e.g. CSF)
np.sum(lohi_wsr_rb<0,2) # underestimate (e.g. GM)

# global
lohi_wsr_g_rb, lohi_wsr_g_p = np.zeros([ncon,nacq,nroi_g]), np.zeros([ncon,nacq,nroi_g])
for c in range(ncon):
    for a in range(nacq):
        for r in range(nroi_g):
            # IDs of subjects which don't contain NaN
            not_nan_id = np.setdiff1d(np.arange(ns),np.where(np.isnan(sseg_hf_g[:,0,c,a,r]))[0])
            # statistics
            lohi_wsr_g_rb[c,a,r] = pg.wilcoxon(sseg_hf_g[:,0,c,a,r][not_nan_id],sseg_ge_g[:,c,r][not_nan_id])['RBC']
            lohi_wsr_g_p[c,a,r] = pg.wilcoxon(sseg_hf_g[:,0,c,a,r][not_nan_id],sseg_ge_g[:,c,r][not_nan_id])['p-val']

# proportions per scan
np.sum(lohi_wsr_g_rb>0,2) # overestimate (e.g. CSF)
np.sum(lohi_wsr_g_rb<0,2) # underestimate (e.g. GM)

# %% percentage difference

# local
lohi_pdiff = np.zeros([ns,ncon,nacq,nroi])
for c in range(ncon):
    for a in range(nacq):
        for r in range(nroi):
            for s in range(ns):
                # statistics
                lohi_pdiff[s,c,a,r] = 100*(sseg_hf[s,0,c,a,r] - sseg_ge[s,c,r])/sseg_ge[s,c,r]

# c, a, r
#plt.hist(np.nanmedian(lohi_pdiff,0)[1,-1,:],30)
np.median(lohi_pdiff,0)[1,-1,:]

np.min(np.nanmedian(lohi_pdiff,0)[1,-1,:])              # min
np.percentile(np.nanmedian(lohi_pdiff,0)[1,-1,:], 25)   # q1
np.median(np.nanmedian(lohi_pdiff,0)[1,-1,:])           # md
np.percentile(np.nanmedian(lohi_pdiff,0)[1,-1,:], 75)   # q3
np.max(np.nanmedian(lohi_pdiff,0)[1,-1,:])              # max

# global
lohi_pdiff_g = np.zeros([ns,ncon,nacq,nroi_g])
for c in range(ncon):
    for a in range(nacq):
        for r in range(nroi_g):
            for s in range(ns):
                # statistics
                lohi_pdiff_g[s,c,a,r] = 100*(sseg_hf_g[s,0,c,a,r] - sseg_ge_g[s,c,r])/sseg_ge_g[s,c,r]

# c, a, r
np.nanmedian(lohi_pdiff_g,0)[1,-1,:]

# %% write percentage difference results

nround = 3                                          # number of decimal places to round
filename = plot_dir+'/lohi_diff/lohi_pdiff.csv'   # output

with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)

    for c in range(ncon):
    
        # header
        writer.writerow(['statistic'] + list(acq[c,:]))
        
        ### Pearson's r
        
        # global median (inner median is across subjects, outer median is across regions)
        writer.writerow(np.concatenate((np.array(['perc. diff. glob Md']),np.round(np.nanmedian(np.nanmedian(lohi_pdiff_g,0),2)[c,:],nround))))
        
        # local median [q1,q3]
        qrow = ['perc. diff. loc Md [Q1, Q3]']
        for a in range(nacq):
            qrow += [f'{round(np.nanmedian(np.nanmedian(lohi_pdiff,0), 2)[c,a],nround)} [{round(np.nanpercentile(np.nanmedian(lohi_pdiff,0), 25, 2)[c,a],nround)},{round(np.nanpercentile(np.nanmedian(lohi_pdiff,0), 75, 2)[c,a],nround)}]']
        writer.writerow(qrow)
        
        # ### Lin's CCC
        
        # # global median
        # writer.writerow(np.concatenate((np.array(['Lin\'s CCC glob Md']),np.round(np.nanmedian(lohi_lin_g, 2)[c,:],nround))))
        
        # # local median [q1,q3]
        # qrow = ['Lin\'s CCC loc Md [Q1, Q3]']
        # for a in range(nacq):
        #     qrow += [f'{round(np.nanmedian(lohi_lin, 2)[c,a],nround)} [{round(np.nanpercentile(lohi_lin, 25, 2)[c,a],nround)},{round(np.nanpercentile(lohi_lin, 75, 2)[c,a],nround)}]']
        # writer.writerow(qrow)
        
        # blank row (between contrasts)
        writer.writerow(' ')

# %% ULF-HF: segmentations - RIGID (takes a long time to run)

lohi_dice_g = np.zeros([ns,ncon,nacq,nroi_g])
lohi_dice = np.zeros([ns,ncon,nacq,nroi])
for s in range(ns):
    print(f'----\nsubject {str(s)}\n----')
    for c in range(ncon):
        print(f'--\ncontrast: {con[c]}\n--')
        for a in range(nacq):
            print(f'acquisition: {acq[c,a]}')
            # paths to files
            #hfc_nii_path = sreg_dir[c]+'/sub-'+sub[s]+'/sub-'+sub[s]+'_ses-HFC_'+acq[c,a]+'_synthseg_to_'+sreg_target[c]+'.nii.gz'
            hfc_nii_path = sreg_dir[c]+'/sub-'+sub[s]+'/sub-'+sub[s]+'_ses-HFC_'+acq[c,a]+'_synthseg_flirt_to_ses-GE_'+sreg_target[c]+'.nii.gz'
            ge_nii_path = sseg_dir[c]+'/sub-'+sub[s]+'/sub-'+sub[s]+'_ses-GE_'+sseg_ge_str[c]+'_synthseg.nii.gz'
            
            # if files exist, load them...
            if (os.path.isfile(hfc_nii_path)) and (os.path.isfile(ge_nii_path)):
                
                # load files
                hfc = np.array(nib.load(hfc_nii_path).dataobj).flatten()
                ge = np.array(nib.load(ge_nii_path).dataobj).flatten()
                
                # dice global 
                for g in range(nroi_g):
                    hfc_g,ge_g = np.zeros(len(hfc)),np.zeros(len(ge))   # initialise empty variable
                    for gr in range(len(roi_g_ind[g])):                 # loop over aggregate indices for current tissue class
                        hfc_g[hfc==seg_id[seg_match][roi_g_ind[g][gr]]] = 1                # fill temporary vector for current class - hfc
                        ge_g[ge==seg_id[seg_match][roi_g_ind[g][gr]]] = 1                # fill temporary vector for current class - hfc
                    lohi_dice_g[s,c,a,g] = 1-dice(hfc_g!=0,ge_g!=0)     # calculate dice
               
                # dice local
                for r in range(nroi):
                    lohi_dice[s,c,a,r] = 1-dice(hfc==seg_id[seg_match][r],ge==seg_id[seg_match][r])
               
            # ...else fill with NaNs
            else:
                lohi_dice_g[s,c,a,:] = np.nan
                lohi_dice[s,c,a,:] = np.nan

np.save(plot_dir+'/lohi_dice/lohi_dice_g.npy', lohi_dice_g)
np.save(plot_dir+'/lohi_dice/lohi_dice.npy', lohi_dice)

# %% load previously calculated Dice results

lohi_dice_g = np.load(plot_dir+'/lohi_dice/lohi_dice_g.npy')
lohi_dice = np.load(plot_dir+'/lohi_dice/lohi_dice.npy')

# %% lo-hi global Dice plots

import ptitprince as pt          # raincloud plots

for c in range(ncon):
    for a in range(nacq):
    
        # IDs of subjects which don't contain NaN 
        not_nan_id = np.setdiff1d(np.arange(ns),np.where(np.isnan(lohi_dice_g[:,c,a,0]))[0]) # g = 0
    
        if len(not_nan_id) != 0:
    
            #fig = plt.figure(figsize=(8, 5))#, layout='compressed') # layout='constrained')
            f, ax = plt.subplots(figsize=(5, 5))
            
            dx = list( np.concatenate(( np.repeat(0,len(not_nan_id)), np.repeat(1,len(not_nan_id)), np.repeat(2,len(not_nan_id)), np.repeat(3,len(not_nan_id)) )) )
            dy = list( lohi_dice_g[not_nan_id,c,a,:].T.flatten() )

            pt.RainCloud(x = dx, y = dy, palette = roi_g_col_lohi, bw = .4, width_viol = 0.5, orient = "v", box_showfliers=False)
            ax.set_xticklabels(roi_g_plt, size=lbs-2); #plt.xticks(fontsize=lbs-1); 
            plt.ylabel('Dice coefficient', size=lbs+4); plt.yticks(fontsize=lbs-5); 
            plt.xlim([-0.65,3.35])
            plt.ylim([0.2,1.0])
            plt.title(acq_plt[c,a], fontsize=lbs+1)
            #f.tight_layout()
            #fig.tight_layout(rect=[0.05, 0.01, 1, 0.95])
            
            #if save_fig: plt.savefig(plot_dir+'/lohi_dice/lohi_dice_'+acq[c,acq_subset_id[a]]+'.png',dpi=600) # ,bbox_inches='tight'
            #if save_fig: plt.savefig(plot_dir+'/lohi_dice/lohi_dice_'+acq[c,a]+'.png',dpi=600) # ,bbox_inches='tight'
            if save_fig: plt.savefig(plot_dir+'/lohi_dice/lohi_dice_'+acq[c,a]+'.svg') # ,bbox_inches='tight'
        
            else:
                continue
    
# %% lo-hi local tissue volume plots (whole-brain "segmentation" plots)

# mean Dice (T1w / T2w)     np.nanmedian(lohi_dice,0)   (2, 5, 98)

# MNI template volume
seg_nii = nib.load(hype_dir+'/mni/mni_synthseg.nii')
seg_vec = np.array(seg_nii.dataobj).flatten()
seg_mask = (seg_vec!=0)*1

# extract parameters from example volume
nvox = np.array(seg_nii.dataobj).size           # number of voxels
nii_shape = np.array(seg_nii.dataobj).shape     # image dimensions
nii_affine = seg_nii.affine                     # affine

for c in range(ncon):
    for a in range(nacq):

        ### Dice
        
        # fill vector
        plot_vec = np.zeros(seg_vec.shape)
        for r in range(nroi):
            plot_vec[seg_vec==seg_id[seg_match][r]] = np.nanmedian(lohi_dice,0)[c,a,r]
        
        # plot seg
        f = plt.figure(figsize=(8, 4))
        cl = [deci_floor(np.min(np.nanmedian(lohi_dice,0)[c,a,:]),1),deci_ceil(np.max(np.nanmedian(lohi_dice,0)[c,a,:]),1)]
        #cl = [0,1]
        plot_nl_image_masked(plot_vec, seg_mask, nii_shape, nii_affine, annotate=False, cmap='plasma_r', clim=cl, cut_coords=np.array([-20,0,20]), draw_cross=True,black_bg=False,display_mode='z',figure=f,output_file=plot_dir+'/lohi_dice/lohi_mni_md_dice_'+acq[c,a]+'.png')

# %% write Dice results

nround = 3                                      # number of decimal places to round
filename = plot_dir+'/lohi_dice/lohi_dice.csv'  # output

with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)

    for c in range(ncon):
    
        # header
        writer.writerow(['statistic'] + list(acq[c,:]))
        
        # global median
        writer.writerow(np.concatenate((np.array(['Md Dice glob Md']),np.round(np.nanmedian(np.nanmedian(lohi_dice_g,0),2)[c,:],nround))))
        
        # local median [q1,q3]
        qrow = ['Md Dice loc Md [Q1, Q3]']
        for i in range(nacq):
            qrow += [f'{round(np.nanmedian(np.nanmedian(lohi_dice,0), 2)[c,i],nround)} [{round(np.nanpercentile(np.nanmedian(lohi_dice,0), 25, 2)[c,i],nround)},{round(np.nanpercentile(np.nanmedian(lohi_dice,0), 75, 2)[c,i],nround)}]']
        writer.writerow(qrow)
        
        # blank row (between contrasts)
        writer.writerow(' ')

# %% lo-hi Dice bias?

lohi_dice_bias_rho, lohi_dice_bias_p = np.zeros([ncon,nacq]), np.zeros([ncon,nacq])
for c in range(ncon):
    for a in range(nacq):
        lohi_dice_bias_rho[c,a] = stats.spearmanr(np.nanmedian(sseg_ge[:,c,:],0),np.nanmedian(lohi_dice,0)[c,a,:])[0]
        lohi_dice_bias_p[c,a] = stats.spearmanr(np.nanmedian(sseg_ge[:,c,:],0),np.nanmedian(lohi_dice,0)[c,a,:])[1]

# %% write lo-hi Dice bias results

nround = 3                                              # number of decimal places to round
filename = plot_dir+'/lohi_dice/lohi_dice_bias.csv'     # output

with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)

    for c in range(ncon):
    
        # header
        writer.writerow(['statistic'] + list(acq[c,:]))
        
        # rho & p
        writer.writerow(np.concatenate((np.array(['Md Dice Spearman\'s rho']),np.round(lohi_dice_bias_rho[c,:],nround))))
        writer.writerow(np.concatenate((np.array(['Md Dice Spearman\'s p']),np.array(['{:g}'.format(float('{:.{p}g}'.format(i, p=nround))) for i in lohi_dice_bias_p[c,:]]))))
        
        # blank row (between contrasts)
        writer.writerow(' ')

# %%

"""
Reliability VS Correspondence
"""

# directory to save figures
if not os.path.isdir(plot_dir+'/trt_vs_lohi'): os.mkdir(plot_dir+'/trt_vs_lohi')

# %% large correlation matrix
# https://stackoverflow.com/questions/57226054/seaborn-correlation-matrix-with-p-values-with-python
# https://stackoverflow.com/questions/52393084/how-to-assign-custom-color-to-masked-cells-in-seaborn-heatmap

from matplotlib import colormaps

c = 1
a = -1

bias_all = np.vstack((trt_icc[c,a,:],
                      np.median(trt_dice[:,c,a,:],0),
                      lohi_r[c,a,:],
                      lohi_lin[c,a,:],
                      np.median(lohi_dice[:,c,a,:],0),
                      np.nanmedian(sseg_ge[:,c,:],0)/1e3,))

hmap_lab = ['reliability ICC','reliability Dice', 'corresp. r', 'corresp. CCC', 'corresp. Dice', 'volume']

rho,p = stats.spearmanr(bias_all.T)

mask = np.zeros_like(rho, dtype=np.bool)
mask[np.tril_indices_from(mask)] = True

cmap = colormaps.get_cmap('Reds') 
cmap.set_bad("lightgrey")

plt.figure(figsize=(7,6))
#sb.heatmap(rho, annot=True, cmap='magma_r', fmt=".2f", linewidths=0.5)
sb.set(font_scale=1.85)
sb.heatmap(rho, mask = mask & (p > 0.05/15), xticklabels=hmap_lab, yticklabels=hmap_lab, annot=True, cmap=cmap, vmin=0, vmax = 1, fmt=".2f", linewidths=0.5, linecolor='black', annot_kws={"size": 18})
plt.savefig(plot_dir+'/trt_vs_lohi/heatmap_'+acq[c,a]+'.png',dpi=600,bbox_inches='tight')

# %%

icc_vs_r_rho, icc_vs_r_p = np.zeros([ncon,nacq]), np.zeros([ncon,nacq])
icc_vs_lin_rho, icc_vs_lin_p = np.zeros([ncon,nacq]), np.zeros([ncon,nacq])
r_vs_lin_rho, r_vs_lin_p = np.zeros([ncon,nacq]), np.zeros([ncon,nacq])
for c in range(ncon):
    for a in range(nacq):
                # ICC vs r
                icc_vs_r_rho[c,a] = stats.spearmanr(trt_icc[c,a,:],lohi_r[c,a,:])[0]
                icc_vs_r_p[c,a] = stats.spearmanr(trt_icc[c,a,:],lohi_r[c,a,:])[1]
                # ICC vs lin
                icc_vs_lin_rho[c,a] = stats.spearmanr(trt_icc[c,a,:],lohi_lin[c,a,:])[0]
                icc_vs_lin_p[c,a] = stats.spearmanr(trt_icc[c,a,:],lohi_lin[c,a,:])[1]
                # r vs Lin
                r_vs_lin_rho[c,a] = stats.spearmanr(lohi_r[c,a,:],lohi_lin[c,a,:])[0]
                r_vs_lin_p[c,a] = stats.spearmanr(lohi_r[c,a,:],lohi_lin[c,a,:])[1]

# %% write lo-hi bias results

nround = 3                                          # number of decimal places to round
filename = plot_dir+'/trt_vs_lohi/trt_vs_lohi.csv'   # output

with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)

    for c in range(ncon):
    
        # header
        writer.writerow(['statistic'] + list(acq[c,:]))
        
        # ICC vs r
        writer.writerow(np.concatenate((np.array(['ICC vs r: Spearman\'s rho']),np.round(icc_vs_r_rho[c,:],nround))))
        writer.writerow(np.concatenate((np.array(['ICC vs r: Spearman\'s p']),np.array(['{:g}'.format(float('{:.{p}g}'.format(i, p=nround))) for i in icc_vs_r_p[c,:]]))))
        
        # ICC vs Lin's CCC
        writer.writerow(np.concatenate((np.array(['ICC vs CCC: Spearman\'s rho']),np.round(icc_vs_lin_rho[c,:],nround))))
        writer.writerow(np.concatenate((np.array(['ICC vs: CCC Spearman\'s p']),np.array(['{:g}'.format(float('{:.{p}g}'.format(i, p=nround))) for i in icc_vs_lin_p[c,:]]))))
        
        # r vs Lin's CCC
        writer.writerow(np.concatenate((np.array(['r vs CCC: Spearman\'s rho']),np.round(r_vs_lin_rho[c,:],nround))))
        writer.writerow(np.concatenate((np.array(['r vs: CCC Spearman\'s p']),np.array(['{:g}'.format(float('{:.{p}g}'.format(i, p=nround))) for i in r_vs_lin_p[c,:]]))))
        
        # blank row (between contrasts)
        writer.writerow(' ')

# %%

sb.set_theme(style='ticks')

# %%

c = 1
a = -1

icc_vs_r_rho[c,a]
icc_vs_lin_rho[c,a]
r_vs_lin_rho[c,a]

# ICC vs r

fig = plt.figure(figsize=(4,4))#, layout='compressed') # layout='constrained')#

x = np.copy(trt_icc[c,a,:])
y = np.copy(lohi_r[c,a,:])

# limits and ticks 
x_lim = [0.95*np.min(x),1.05*np.max(x)]
y_lim = [0.95*np.min(y),1.05*np.max(y)]

# plot
sb.regplot(x=x, y=y, ci=95, scatter_kws={"color": "grey"}, line_kws={"color": "black"}) # line of best fit + CI
plt.xticks(fontsize=axs-2); plt.xlim(x_lim);  # plt.xticks(fontsize=axs-1);        # axis limits and ticks
plt.yticks(fontsize=axs-2); plt.ylim(y_lim);  # plt.yticks(fontsize=axs-1); 

plt.xlabel('reliability (ICC)',fontsize=lbs)
plt.ylabel('correspondence (r)',fontsize=lbs)
plt.title(r'$\rho$ = '+str(round(icc_vs_r_rho[c,a],2))+', P < 10$^{-10}$',fontsize=lbs-1,pad=13)

if save_fig: 
    plt.savefig(plot_dir+'/trt_vs_lohi/icc_vs_r_'+acq[c,a]+'.png',dpi=600) # ,bbox_inches='tight'
    plt.savefig(plot_dir+'/trt_vs_lohi/icc_vs_r_'+acq[c,a]+'.svg')
    
# ICC vs Lin

fig = plt.figure(figsize=(4,4))#, layout='compressed') # layout='constrained')#

x = np.copy(trt_icc[c,a,:])
y = np.copy(lohi_lin[c,a,:])

# limits and ticks 
x_lim = [0.95*np.min(x),1.05*np.max(x)]
y_lim = [0.8*np.min(y),1.05*np.max(y)]

# plot
sb.regplot(x=x, y=y, ci=95, scatter_kws={"color": "grey"}, line_kws={"color": "black"}) # line of best fit + CI
plt.xticks(fontsize=axs-2); plt.xlim(x_lim);  # plt.xticks(fontsize=axs-1);        # axis limits and ticks
plt.yticks(fontsize=axs-2); plt.ylim(y_lim);  # plt.yticks(fontsize=axs-1); 

plt.xlabel('reliability (ICC)',fontsize=lbs)
plt.ylabel('correspondence (CCC)',fontsize=lbs)
plt.title(r'$\rho$ = '+str(round(icc_vs_lin_rho[c,a],2))+', P < 10$^{-10}$',fontsize=lbs-1,pad=13)

if save_fig: 
    plt.savefig(plot_dir+'/trt_vs_lohi/icc_vs_lin_'+acq[c,a]+'.png',dpi=600) # ,bbox_inches='tight'
    plt.savefig(plot_dir+'/trt_vs_lohi/icc_vs_lin_'+acq[c,a]+'.svg')
    
# %%    
    
# trt Dice vs lohi Dice

fig = plt.figure(figsize=(4,4))#, layout='compressed') # layout='constrained')#

x = np.copy(np.median(trt_dice[:,c,a,:],0))
y = np.copy(np.median(lohi_dice[:,c,a,:],0))

# limits and ticks 
x_lim = [0.95*np.min(x),1.05*np.max(x)]
y_lim = [0.8*np.min(y),1.05*np.max(y)]

# plot
sb.regplot(x=x, y=y, ci=95, scatter_kws={"color": "grey"}, line_kws={"color": "black"}) # line of best fit + CI
plt.xticks(fontsize=axs-2); plt.xlim(x_lim);  # plt.xticks(fontsize=axs-1);        # axis limits and ticks
plt.yticks(fontsize=axs-2); plt.ylim(y_lim);  # plt.yticks(fontsize=axs-1); 

rho = stats.spearmanr(np.median(trt_dice[:,c,a,:],0),np.median(lohi_dice[:,c,a,:],0))[0]
p = stats.spearmanr(np.median(trt_dice[:,c,a,:],0),np.median(lohi_dice[:,c,a,:],0))[1]

plt.xlabel('reliability (Dice)',fontsize=lbs)
plt.ylabel('correspondence (Dice)',fontsize=lbs)
plt.title(r'$\rho$ = '+str(round(rho,2))+', P < 10$^{-10}$',fontsize=lbs-1,pad=13)

if save_fig: 
    plt.savefig(plot_dir+'/trt_vs_lohi/trt_dice_vs_lohi_dice_'+acq[c,a]+'.png',dpi=600) # ,bbox_inches='tight'
    plt.savefig(plot_dir+'/trt_vs_lohi/trt_dice_vs_lohi_dice_'+acq[c,a]+'.svg')

# %% volume bias plots

# ICC vs vol

fig = plt.figure(figsize=(4,4))#, layout='compressed') # layout='constrained')#

#x = np.copy(trt_icc[c,a,:])
x = np.copy(np.nanmedian(sseg_ge[:,c,:],0)/1e3)
y = np.copy(trt_icc[c,a,:])

# limits and ticks 
x_lim = [-2,28]
y_lim = [0.95*np.min(y),1.05*np.max(y)]

# plot
sb.regplot(x=x, y=y, ci=95, scatter_kws={"color": "grey"}, fit_reg=False)# line_kws={"color": "black"}) # line of best fit + CI
#sb.scatterplot(x, y, ci=95, scatter_kws={"color": "grey"})#, line_kws={"color": "black"}) # line of best fit + CI
plt.xticks(fontsize=axs-2); plt.xlim(x_lim);  # plt.xticks(fontsize=axs-1);        # axis limits and ticks
plt.yticks(fontsize=axs-2); plt.ylim(y_lim);  # plt.yticks(fontsize=axs-1); 

plt.xlabel('volume (cm$^3$)',fontsize=lbs)
plt.ylabel('reliability (ICC)',fontsize=lbs)
plt.title(r'$\rho$ = '+str(round(trt_icc_bias_rho[c,a],2))+', P = '+'{:0.2e}'.format(trt_icc_bias_p[c,a]),fontsize=lbs-1,pad=13)

if save_fig: 
    plt.savefig(plot_dir+'/trt_vs_lohi/icc_vs_vol_'+acq[c,a]+'.png',dpi=600) # ,bbox_inches='tight'
    plt.savefig(plot_dir+'/trt_vs_lohi/icc_vs_vol_'+acq[c,a]+'.svg')

x_lim = [-10,1.05*np.max(x)]; plt.xlim(x_lim);
if save_fig: plt.savefig(plot_dir+'/trt_vs_lohi/icc_vs_vol_'+acq[c,a]+'_uncropped.svg')


# %% volume segmentation plot

# MNI template volume
seg_nii = nib.load(hype_dir+'/mni/mni_synthseg.nii')
seg_vec = np.array(seg_nii.dataobj).flatten()
seg_mask = (seg_vec!=0)*1

# extract parameters from example volume
nvox = np.array(seg_nii.dataobj).size           # number of voxels
nii_shape = np.array(seg_nii.dataobj).shape     # image dimensions
nii_affine = seg_nii.affine                     # affine
        
# fill vector
plot_vec = np.zeros(seg_vec.shape)
for r in range(nroi):
    # if sum(seg_vec==seg_id[seg_match][i])==0:
    #     print(f'contrast {c}, acquisition {a}, ROI {i} empty')
    plot_vec[seg_vec==seg_id[seg_match][r]] = np.nanmedian(sseg_ge[:,1,r])/1e3

# plot seg
f = plt.figure(figsize=(8, 4))
cl = [0,30] #[ deci_floor(np.min(np.nanmedian(sseg_ge[:,1,:],0)/1e3),1) , deci_ceil(np.max(np.nanmedian(sseg_ge[:,c,:],0)/1e3),1) ]
plot_nl_image_masked(plot_vec, seg_mask, nii_shape, nii_affine, annotate=False, cmap='cividis_r', clim=cl, cut_coords=np.array([-20,0,20]), draw_cross=True,black_bg=False,display_mode='z',figure=f,output_file=plot_dir+'/trt_vs_lohi/vol_mni_icc_t2w.png')

# %%

"""
Tissue volume VS Age
"""

# directory to save figures
if not os.path.isdir(plot_dir+'/age'): os.mkdir(plot_dir+'/age')

# %% multiple regression function
# https://stackoverflow.com/questions/11479064/multiple-linear-regression-in-python

import statsmodels.api as sm

def reg_m(y, x):
    x = np.array(x).T
    x = sm.add_constant(x) 
    results = sm.OLS(endog=y, exog=x).fit()
    return results

# print reg_m(y, x).summary()

# %% calculate multiple regression of Hyperfine (HFC) volume as a function of age, covarying for sex

# local
age_lo_rsq = np.zeros([ncon,nacq,nroi])
age_lo_t = np.zeros([ncon,nacq,nroi])
age_lo_p = np.zeros([ncon,nacq,nroi])
for c in range(ncon):
    print(f'--\ncontrast: {con[c]}\n--')
    for a in range(nacq):
        print(f'acquisition: {acq[c,a]}')
        for r in range(nroi):
            # IDs of subjects which don't contain NaN (in the HFC / first = 0 session only, i.e. sseg_hf[:,0,c,a,r])
            not_nan_id = np.setdiff1d(np.arange(ns),np.where(np.isnan(sseg_hf[:,0,c,a,r]))[0])
            # fit model
            reg_res = reg_m(sseg_hf[:,0,c,a,r][not_nan_id], [age[not_nan_id],sex_male[not_nan_id]])
            #reg_stat = reg_res.summary()
            # extract statistics
            age_lo_rsq[c,a,r] = reg_res.rsquared
            age_lo_t[c,a,r] = reg_res.tvalues[1]
            age_lo_p[c,a,r] = reg_res.pvalues[1]    # p-value for the effect of age
            
# global
age_lo_rsq_g = np.zeros([ncon,nacq,nroi_g])
age_lo_t_g = np.zeros([ncon,nacq,nroi_g])
age_lo_p_g = np.zeros([ncon,nacq,nroi_g])
for c in range(ncon):
    print(f'--\ncontrast: {con[c]}\n--')
    for a in range(nacq):
        print(f'acquisition: {acq[c,a]}')
        for r in range(nroi_g):
            # IDs of subjects which don't contain NaN (in the HFC / first = 0 session only, i.e. sseg_hf[:,0,c,a,r])
            not_nan_id = np.setdiff1d(np.arange(ns),np.where(np.isnan(sseg_hf[:,0,c,a,r]))[0])
            # fit model
            reg_res = reg_m(sseg_hf_g[:,0,c,a,r][not_nan_id], [age[not_nan_id],sex_male[not_nan_id]])
            #reg_stat = reg_res.summary()
            # extract statistics
            age_lo_rsq_g[c,a,r] = reg_res.rsquared
            age_lo_t_g[c,a,r] = reg_res.tvalues[1]
            age_lo_p_g[c,a,r] = reg_res.pvalues[1]    # p-value for the effect of age

# %% as above, for high-field (GE) data

# local
age_hi_rsq = np.zeros([ncon,nroi])
age_hi_t = np.zeros([ncon,nroi])
age_hi_p = np.zeros([ncon,nroi])
for c in range(ncon):
    #print(f'--\ncontrast: {con[c]}\n--')
    for r in range(nroi):
        # fit model
        reg_res = reg_m(sseg_ge[:,c,r], [age[not_nan_id],sex_male[not_nan_id]])
        #reg_stat = reg_res.summary()
        # extract statistics
        age_hi_rsq[c,r] = reg_res.rsquared
        age_hi_t[c,r] = reg_res.tvalues[1]
        age_hi_p[c,r] = reg_res.pvalues[1]  # p-value for the effect of age

# global
age_hi_rsq_g = np.zeros([ncon,nroi_g])
age_hi_t_g = np.zeros([ncon,nroi_g])
age_hi_p_g = np.zeros([ncon,nroi_g])
for c in range(ncon):
    #print(f'--\ncontrast: {con[c]}\n--')
    for r in range(nroi_g):
        # fit model
        reg_res = reg_m(sseg_ge_g[:,c,r], [age[not_nan_id],sex_male[not_nan_id]])
        #reg_stat = reg_res.summary()
        # extract statistics
        age_hi_rsq_g[c,r] = reg_res.rsquared
        age_hi_t_g[c,r] = reg_res.tvalues[1]
        age_hi_p_g[c,r] = reg_res.pvalues[1]  # p-value for the effect of age

# %% correlations between r-squared and t-stat vectors (within contrasts)

age_rsq_r, age_t_r = np.zeros([ncon,nacq]), np.zeros([ncon,nacq])
age_rsq_lin, age_t_lin = np.zeros([ncon,nacq]), np.zeros([ncon,nacq])
for c in range(ncon):
    for a in range(nacq):
        # r
        age_rsq_r[c,a] = stats.pearsonr(age_lo_rsq[c,a,:],age_hi_rsq[c,:])[0]
        age_t_r[c,a] = stats.pearsonr(age_lo_t[c,a,:],age_hi_t[c,:])[0]
        # lin
        age_rsq_lin[c,a] = lin(age_lo_rsq[c,a,:],age_hi_rsq[c,:])
        age_t_lin[c,a] = lin(age_lo_t[c,a,:],age_hi_t[c,:])


# %% age scatterplots - global - ultra-low-field

sb.set_style(style='ticks')

for c in range(ncon):
    for a in range(nacq):

        # IDs of subjects which don't contain NaN in the HFC session
        not_nan_id = np.setdiff1d(np.arange(ns),np.where(np.isnan(sseg_hf_g[:,0,c,a,0]))[0]) # ss = 0, g = 0

        if len(not_nan_id) != 0:
        
            fig = plt.figure(figsize=(13, 4.5))
        
            for g in range(nroi_g):
                
                plt.subplot(1,nroi_g,g+1)#, aspect=0.35)
    
                x = np.copy(age[not_nan_id])
                y = np.copy(sseg_hf_g[:,0,c,a,g][not_nan_id]/1e3) 
                
                # # plot set-up
                age_x_ticks = np.arange(10, 75, 10)
                age_x_lim = [15, 75]
                y_lim = [0.85*np.min(y),1.1*np.max(y)]
                
                # statistics
                reg_res = reg_m(sseg_hf_g[:,0,c,a,g][not_nan_id]/1e3, [age[not_nan_id],sex_male[not_nan_id]])
                
                # plot
                sb.regplot(x=x, y=y, ci=95, scatter_kws={"color": "grey"}, line_kws={"color": "black"}) # line of best fit + CI
                plt.xticks(fontsize=axs-2); plt.xlim(age_x_lim);                                       # axis limits and ticks
                plt.yticks(fontsize=axs-2); plt.ylim(y_lim); 
    
                # title and save
                plt.title('r$^2$ = '+str(round(reg_res.rsquared,2))+' \n t = '+str(round(reg_res.tvalues[1],2))+'; p = '+str('{:g}'.format(float('{:.{p}g}'.format(reg_res.pvalues[1], p=2)))),fontsize=lbs-2,pad=13, linespacing=1.65)

            fig.tight_layout(rect=[0.05, 0.07, 1, 0.88])
            fig.supxlabel('age (years)',fontsize=lbs)
            fig.supylabel('64mT volume (cm$^3$)',fontsize=lbs)
            
            if save_fig: 
                plt.savefig(plot_dir+'/age/age_hfc_'+acq[c,a]+'.png',dpi=600) # ,bbox_inches='tight'
            
            else:
                continue

# %% age scatterplots - global - high-field

for c in range(ncon):

    # IDs of subjects which don't contain NaN in the GE session
    not_nan_id = np.setdiff1d(np.arange(ns),np.where(np.isnan(sseg_ge_g[:,c,0]))[0]) # ROI 0

    if len(not_nan_id) != 0:
    
        fig = plt.figure(figsize=(13, 4.5))
    
        for g in range(nroi_g):
            
            plt.subplot(1,nroi_g,g+1)#, aspect=0.35)

            x = np.copy(age[not_nan_id])
            y = np.copy(sseg_ge_g[:,c,g][not_nan_id]/1e3)
            
            # # plot set-up
            age_x_ticks = np.arange(10, 75, 10)
            age_x_lim = [15, 75]
            y_lim = [0.75*np.min(y),1.2*np.max(y)]
            
            # statistics
            reg_res = reg_m(sseg_ge_g[:,c,g][not_nan_id]/1e3, [age[not_nan_id],sex_male[not_nan_id]])

            # plot
            sb.regplot(x=x, y=y, ci=95, scatter_kws={"color": "grey"}, line_kws={"color": "black"}) # line of best fit + CI
            plt.xticks(fontsize=axs-2); plt.xlim(age_x_lim);                                       # axis limits and ticks
            plt.yticks(fontsize=axs-2); plt.ylim(y_lim); 

            # title and save
            plt.title('r$^2$ = '+str(round(reg_res.rsquared,2))+' \n t = '+str(round(reg_res.tvalues[1],2))+'; p = '+str('{:g}'.format(float('{:.{p}g}'.format(reg_res.pvalues[1], p=2)))),fontsize=lbs-2,pad=13, linespacing=1.65)

        fig.tight_layout(rect=[0.05, 0.07, 1, 0.88])
        fig.supxlabel('age (years)',fontsize=lbs)
        fig.supylabel('3T volume (cm$^3$)',fontsize=lbs)
        
        if save_fig: 
            plt.savefig(plot_dir+'/age/age_ge_'+acq[c,a]+'.png',dpi=600) # ,bbox_inches='tight'
        
        else:
            continue

# %% plot age results (whole-brain "segmentation" plots)

# r-squared (T1w / T2w)     age_lo_rsq      (2, 5, 98)
# t-stat (T1w / T2w)        age_lo_t        (2, 5, 98)

# MNI template volume
seg_nii = nib.load(hype_dir+'/mni/mni_synthseg.nii')
seg_vec = np.array(seg_nii.dataobj).flatten()
seg_mask = (seg_vec!=0)*1

# extract parameters from example volume
nvox = np.array(seg_nii.dataobj).size           # number of voxels
nii_shape = np.array(seg_nii.dataobj).shape     # image dimensions
nii_affine = seg_nii.affine                     # affine

for c in range(ncon):
    
    ##### ultra-low-field
    for a in range(nacq):
        
        ### r-squared
        
        # fill vector
        plot_vec = np.zeros(seg_vec.shape)
        for r in range(nroi):
            plot_vec[seg_vec==seg_id[seg_match][r]] = age_lo_rsq[c,a,r]
    
        # plot seg
        f = plt.figure(figsize=(8, 4))
        cl = [0,0.8]
        plot_nl_image_masked(plot_vec, seg_mask, nii_shape, nii_affine, annotate=False, cmap='plasma_r', clim=cl, cut_coords=np.array([-20,0,20]), draw_cross=True,black_bg=False,display_mode='z',figure=f,output_file=plot_dir+'/age/age_mni_rsq_'+acq[c,a]+'.png')

        ### t-statistic
        
        # fill vector
        plot_vec = np.zeros(seg_vec.shape)
        for r in range(nroi):
            plot_vec[seg_vec==seg_id[seg_match][r]] = age_lo_t[c,a,r]
        
        # plot seg
        f = plt.figure(figsize=(8, 4))
        cl = [-np.max([abs(deci_floor(np.min(age_lo_t[c,a,:]),1)),abs(deci_ceil(np.max(age_lo_t[c,a,:]),1))]),np.max([abs(deci_floor(np.min(age_lo_t[c,a,:]),1)),abs(deci_ceil(np.max(age_lo_t[c,a,:]),1))])]
        plot_nl_image_masked(plot_vec, seg_mask, nii_shape, nii_affine, annotate=False, cmap='coolwarm', clim=cl, cut_coords=np.array([-20,0,20]), draw_cross=True,black_bg=False,display_mode='z',figure=f,output_file=plot_dir+'/age/age_mni_tstat_'+acq[c,a]+'.png')

    ##### high-field
    
    ### r-squared
    
    # fill vector
    plot_vec = np.zeros(seg_vec.shape)
    for r in range(nroi):
        plot_vec[seg_vec==seg_id[seg_match][r]] = age_hi_rsq[c,r]
    
    # plot seg
    f = plt.figure(figsize=(8, 4))
    cl = [0,0.8]
    plot_nl_image_masked(plot_vec, seg_mask, nii_shape, nii_affine, annotate=False, cmap='plasma_r', clim=cl, cut_coords=np.array([-20,0,20]), draw_cross=True,black_bg=False,display_mode='z',figure=f,output_file=plot_dir+'/age/age_mni_rsq_GE_'+sseg_ge_str[c]+'.png')

    ### t-statistic
    
    # fill vector
    plot_vec = np.zeros(seg_vec.shape)
    for r in range(nroi):
        plot_vec[seg_vec==seg_id[seg_match][r]] = age_hi_t[c,r]
    
    # plot seg
    f = plt.figure(figsize=(8, 4))
    cl = [-np.max([abs(deci_floor(np.min(age_hi_t[c,:]),1)),abs(deci_ceil(np.max(age_hi_t[c,:]),1))]),np.max([abs(deci_floor(np.min(age_hi_t[c,:]),1)),abs(deci_ceil(np.max(age_hi_t[c,:]),1))])]
    plot_nl_image_masked(plot_vec, seg_mask, nii_shape, nii_affine, annotate=False, cmap='coolwarm', clim=cl, cut_coords=np.array([-20,0,20]), draw_cross=True,black_bg=False,display_mode='z',figure=f,output_file=plot_dir+'/age/age_mni_tstat_GE_'+sseg_ge_str[c]+'.png')

# %% as above, t-stat only, with fixed colormap across hi and lo

# MNI template volume
seg_nii = nib.load(hype_dir+'/mni/mni_synthseg.nii')
seg_vec = np.array(seg_nii.dataobj).flatten()
seg_mask = (seg_vec!=0)*1

# extract parameters from example volume
nvox = np.array(seg_nii.dataobj).size           # number of voxels
nii_shape = np.array(seg_nii.dataobj).shape     # image dimensions
nii_affine = seg_nii.affine                     # affine

c = 1
a = -1

### t-statistic

# fill vector
plot_vec = np.zeros(seg_vec.shape)
for r in range(nroi):
    plot_vec[seg_vec==seg_id[seg_match][r]] = age_lo_t[c,a,r]

# plot seg
f = plt.figure(figsize=(8, 4))
cl = [-np.max([abs(deci_floor(np.min(age_hi_t[c,:]),1)),abs(deci_ceil(np.max(age_hi_t[c,:]),1))]),np.max([abs(deci_floor(np.min(age_hi_t[c,:]),1)),abs(deci_ceil(np.max(age_hi_t[c,:]),1))])]
plot_nl_image_masked(plot_vec, seg_mask, nii_shape, nii_affine, annotate=False, cmap='coolwarm', clim=cl, cut_coords=np.array([-20,0,20]), draw_cross=True,black_bg=False,display_mode='z',figure=f,output_file=plot_dir+'/age/colorbar_age_mni_tstat_'+acq[c,a]+'.png')

##### high-field

### t-statistic

# fill vector
plot_vec = np.zeros(seg_vec.shape)
for r in range(nroi):
    plot_vec[seg_vec==seg_id[seg_match][r]] = age_hi_t[c,r]

# plot seg
f = plt.figure(figsize=(8, 4))
cl = [-np.max([abs(deci_floor(np.min(age_hi_t[c,:]),1)),abs(deci_ceil(np.max(age_hi_t[c,:]),1))]),np.max([abs(deci_floor(np.min(age_hi_t[c,:]),1)),abs(deci_ceil(np.max(age_hi_t[c,:]),1))])]
plot_nl_image_masked(plot_vec, seg_mask, nii_shape, nii_affine, annotate=False, cmap='coolwarm', clim=cl, cut_coords=np.array([-20,0,20]), draw_cross=True,black_bg=False,display_mode='z',figure=f,output_file=plot_dir+'/age/colorbar_age_mni_tstat_GE_'+sseg_ge_str[c]+'.png')

# %% write age results - ultra-low-field ("lo")

nround = 3                                          # number of decimal places to round
filename = plot_dir+'/age/age.csv'      # output

with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)

    for c in range(ncon):
    
        # header
        writer.writerow(['statistic'] + list(acq[c,:]) + ['GE_'+sseg_ge_str[c]])
        
        ### r-squared
        
        # local median [q1,q3]
        qrow = ['r-squared loc Md [Q1, Q3]']
        for a in range(nacq):
            qrow += [f'{round(np.nanmedian(age_lo_rsq, 2)[c,a],nround)} [{round(np.nanpercentile(age_lo_rsq, 25, 2)[c,a],nround)},{round(np.nanpercentile(age_lo_rsq, 75, 2)[c,a],nround)}]']   # low
        qrow += [f'{round(np.nanmedian(age_hi_rsq, 1)[c],nround)} [{round(np.nanpercentile(age_hi_rsq, 25, 1)[c],nround)},{round(np.nanpercentile(age_hi_rsq, 75, 1)[c],nround)}]']             # high
        writer.writerow(qrow)
        
        ### t-statistic
        
        # local median [q1,q3]
        qrow = ['t-stat loc Md [Q1, Q3]']
        for a in range(nacq):
            qrow += [f'{round(np.nanmedian(age_lo_t, 2)[c,a],nround)} [{round(np.nanpercentile(age_lo_t, 25, 2)[c,a],nround)},{round(np.nanpercentile(age_lo_t, 75, 2)[c,a],nround)}]']         # low
        qrow += [f'{round(np.nanmedian(age_hi_t, 1)[c],nround)} [{round(np.nanpercentile(age_hi_t, 25, 1)[c],nround)},{round(np.nanpercentile(age_hi_t, 75, 1)[c],nround)}]']                   # high
        writer.writerow(qrow)
        
        # blank row (between contrasts)
        writer.writerow(' ')

# %%

"""
Min / Md / Max - ICC / r / CCC
"""

if not os.path.isdir(plot_dir+'/min_md_max'): os.mkdir(plot_dir+'/min_md_max')
for i in ['icc','r','lin']: 
    if not os.path.isdir(plot_dir+'/min_md_max/'+i): os.mkdir(plot_dir+'/min_md_max/'+i)

# %%

# Min / Md / Max plots

# sseg_hf = np.zeros([ns,nses,ncon,nacq,nroi])

c = 1   # contrast: T2w
a = -1  # acquisition: mrr

# %% ICC plots

roi_plot_str = ['min','md','max']
nroi_plot = len(roi_plot_str)

# data to plot
plot_data = trt_icc[c,a,:]

# Plot ROIs
roi_plot_min = np.where(plot_data==np.min(plot_data))[0][0]
roi_plot_md = np.where(plot_data==np.percentile(plot_data,50,interpolation='nearest'))[0][0]
roi_plot_max = np.where(plot_data==np.max(plot_data))[0][0]

roi_plot = [roi_plot_min,roi_plot_md,roi_plot_max]
#roi_id = np.where(seg_data[n]==np.min(seg_data[n]))[0]

for r in range(nroi_plot):
    
    # ROI
    roi_nm = vol[vol_match][roi_plot[r]]
    
    # IDs of subjects which don't contain NaN in either of the two sessions (the sum is taken over sessions)
    not_nan_id = np.setdiff1d(np.arange(ns),np.where(np.isnan(np.sum(sseg_hf[:,:,c,a,roi_plot[r]],1)))[0])

    # T-RT
    x = np.copy(sseg_hf[not_nan_id,0,c,a,roi_plot[r]]/1e3)
    y = np.copy(sseg_hf[not_nan_id,1,c,a,roi_plot[r]]/1e3)
    
    # limits and ticks
    ax_range = np.max([np.max(x)-np.min(x),np.max(y)-np.min(y)])
    ax_min = np.min([np.min(x),np.min(y)]) -0.1*ax_range #0.9*np.min([np.min(x),np.min(y)])
    ax_max = np.max([np.max(x),np.max(y)]) +0.1*ax_range #1.05*np.max([np.max(x),np.max(y)])
    ax_lim = [ax_min, ax_max]
    #ax_ticks = np.arange(round(ax_min), round(ax_max)+1, np.ceil(math.log10(ax_max)))
    # plot
    plt.figure(figsize=[5,5]); plt.axis('square')                        # plot set-up
    plt.locator_params(nbins=6)
    plt.plot([ax_lim[0]-1,ax_lim[1]+1], [ax_lim[0]-1,ax_lim[1]+1], ls='--', c='gray')     # plot dashed identity line
    sb.regplot(x=x, y=y, ci=95, scatter_kws={"color": roi_g_col_trt[1]}, line_kws={"color": "black"})  # line of best fit + CI
    plt.xlim(ax_lim); plt.xticks(fontsize=axs+3)    # axis limits and ticks
    plt.ylim(ax_lim); plt.yticks(fontsize=axs+3)
    if save_fig: plt.savefig(plot_dir+'/min_md_max/icc/icc_'+roi_plot_str[r]+'_'+roi_nm.replace('.','').replace(' ','-')+'_nolabel.svg',bbox_inches='tight')
    plt.xlabel('64mT CNS vol. (cm$^3$)',fontsize=lbs,labelpad=10)          # labels
    plt.ylabel('64mT ENIC vol. (cm$^3$)',fontsize=lbs,labelpad=10)
    plt.title(roi_nm+'; ICC = '+str(round(trt_icc[c,a,roi_plot[r]],2)),fontsize=lbs+2,pad=13) #+'; r = '+str(round(trt_r[-1,roi_ad_id_vol[0]],2)))
    if save_fig: plt.savefig(plot_dir+'/min_md_max/icc/icc_'+roi_plot_str[r]+'_'+roi_nm.replace('.','').replace(' ','-')+'.png',bbox_inches='tight',dpi=600)

# %% r plots

roi_plot_str = ['min','md','max']
nroi_plot = len(roi_plot_str)

# data to plot
plot_data = lohi_r[c,a,:]

# Plot ROIs
roi_plot_min = np.where(plot_data==np.min(plot_data))[0][0]
roi_plot_md = np.where(plot_data==np.percentile(plot_data,50,interpolation='nearest'))[0][0]
roi_plot_max = np.where(plot_data==np.max(plot_data))[0][0]

roi_plot = [roi_plot_min,roi_plot_md,roi_plot_max]
#roi_id = np.where(seg_data[n]==np.min(seg_data[n]))[0]

for r in range(nroi_plot):
    
    # ROI
    roi_nm = vol[vol_match][roi_plot[r]]
    
    # IDs of subjects which don't contain NaN in the HFC session
    not_nan_id = np.setdiff1d(np.arange(ns),np.where(np.isnan(sseg_hf[:,0,c,a,roi_plot[r]]))[0])
    
    # T-RT
    x = np.copy(sseg_hf[not_nan_id,0,c,a,roi_plot[r]]/1e3)
    y = np.copy(sseg_ge[not_nan_id,c,roi_plot[r]]/1e3)
    
    # limits and ticks
    ax_range = np.max([np.max(x)-np.min(x),np.max(y)-np.min(y)])
    ax_min = np.min([np.min(x),np.min(y)]) -0.1*ax_range #0.9*np.min([np.min(x),np.min(y)])
    ax_max = np.max([np.max(x),np.max(y)]) +0.1*ax_range #1.05*np.max([np.max(x),np.max(y)])
    ax_lim = [ax_min, ax_max]
    #ax_ticks = np.arange(round(ax_min), round(ax_max)+1, np.ceil(math.log10(ax_max)))
    # plot
    plt.figure(figsize=[5,5]); plt.axis('square')                        # plot set-up
    plt.locator_params(nbins=6)
    plt.plot([ax_lim[0]-1,ax_lim[1]+1], [ax_lim[0]-1,ax_lim[1]+1], ls='--', c='gray')     # plot dashed identity line
    sb.regplot(x=x, y=y, ci=95, scatter_kws={"color":  roi_g_col_lohi[2]}, line_kws={"color": "black"})  # line of best fit + CI
    plt.xlim(ax_lim); plt.xticks(fontsize=axs+3)    # axis limits and ticks
    plt.ylim(ax_lim); plt.yticks(fontsize=axs+3)
    if save_fig: plt.savefig(plot_dir+'/min_md_max/r/r_'+roi_plot_str[r]+'_'+roi_nm.replace('.','').replace(' ','-')+'_nolabel.svg',bbox_inches='tight')
    plt.xlabel('64mT vol. (cm$^3$)',fontsize=lbs,labelpad=10)          # labels
    plt.ylabel('3T vol. (cm$^3$)',fontsize=lbs,labelpad=10)
    # Pearson r
    pr,pp = stats.pearsonr(x,y)
    ccc = lin(x,y)
    # title and save
    plt.title(roi_nm+'; r = '+str(round(pr,2))+'; CCC = '+str(round(ccc,2)),fontsize=lbs+2,pad=13)
    if save_fig: plt.savefig(plot_dir+'/min_md_max/r/r_'+roi_plot_str[r]+'_'+roi_nm.replace('.','').replace(' ','-')+'.png',bbox_inches='tight',dpi=600)

# %% Lin plots

roi_plot_str = ['min','md','max']
nroi_plot = len(roi_plot_str)

# data to plot
plot_data = lohi_lin[c,a,:]

# Plot ROIs
roi_plot_min = np.where(plot_data==np.min(plot_data))[0][0]
roi_plot_md = np.where(plot_data==np.percentile(plot_data,50,interpolation='nearest'))[0][0]
roi_plot_max = np.where(plot_data==np.max(plot_data))[0][0]

roi_plot = [roi_plot_min,roi_plot_md,roi_plot_max]
#roi_id = np.where(seg_data[n]==np.min(seg_data[n]))[0]

for r in range(nroi_plot):
    
    # ROI
    roi_nm = vol[vol_match][roi_plot[r]]
    
    # IDs of subjects which don't contain NaN in the HFC session
    not_nan_id = np.setdiff1d(np.arange(ns),np.where(np.isnan(sseg_hf[:,0,c,a,roi_plot[r]]))[0])
    
    # T-RT
    x = np.copy(sseg_hf[not_nan_id,0,c,a,roi_plot[r]]/1e3)
    y = np.copy(sseg_ge[not_nan_id,c,roi_plot[r]]/1e3)
    
    # limits and ticks
    ax_range = np.max([np.max(x)-np.min(x),np.max(y)-np.min(y)])
    ax_min = np.min([np.min(x),np.min(y)]) -0.1*ax_range #0.9*np.min([np.min(x),np.min(y)])
    ax_max = np.max([np.max(x),np.max(y)]) +0.1*ax_range #1.05*np.max([np.max(x),np.max(y)])
    ax_lim = [ax_min, ax_max]
    #ax_ticks = np.arange(round(ax_min), round(ax_max)+1, np.ceil(math.log10(ax_max)))
    # plot
    plt.figure(figsize=[5,5]); plt.axis('square')                        # plot set-up
    plt.locator_params(nbins=6)
    plt.plot([ax_lim[0]-1,ax_lim[1]+1], [ax_lim[0]-1,ax_lim[1]+1], ls='--', c='gray')     # plot dashed identity line
    sb.regplot(x=x, y=y, ci=95, scatter_kws={"color":  roi_g_col_lohi[2]}, line_kws={"color": "black"})  # line of best fit + CI
    plt.xlim(ax_lim); plt.xticks(fontsize=axs+3)    # axis limits and ticks
    plt.ylim(ax_lim); plt.yticks(fontsize=axs+3)
    if save_fig: plt.savefig(plot_dir+'/min_md_max/lin/lin_'+roi_plot_str[r]+'_'+roi_nm.replace('.','').replace(' ','-')+'_nolabel.svg',bbox_inches='tight')
    plt.xlabel('64mT vol. (cm$^3$)',fontsize=lbs,labelpad=10)          # labels
    plt.ylabel('3T vol. (cm$^3$)',fontsize=lbs,labelpad=10)
    # Pearson r
    pr,pp = stats.pearsonr(x,y)
    ccc = lin(x,y)
    # title and save
    plt.title(roi_nm+'; r = '+str(round(pr,2))+'; CCC = '+str(round(ccc,2)),fontsize=lbs+2,pad=13)
    if save_fig: plt.savefig(plot_dir+'/min_md_max/lin/lin_'+roi_plot_str[r]+'_'+roi_nm.replace('.','').replace(' ','-')+'.png',bbox_inches='tight',dpi=600)

# %%

"""
Min / Md / Max - Dice
"""

if not os.path.isdir(plot_dir+'/min_md_max'): os.mkdir(plot_dir+'/min_md_max')
for i in ['dice_trt','dice_lohi']: 
    if not os.path.isdir(plot_dir+'/min_md_max/'+i): os.mkdir(plot_dir+'/min_md_max/'+i)

# %%

# Min / Md / Max plots

# sseg_hf = np.zeros([ns,nses,ncon,nacq,nroi])

c = 1   # contrast: T2w
a = -1  # acquisition: mrr

# %% Dice trt plots

roi_plot_str = ['min','md','max']
nroi_plot = len(roi_plot_str)

# data to plot
plot_data = np.median(trt_dice[:,c,a,:],0)

# Plot ROIs
roi_plot_min = np.where(plot_data==np.min(plot_data))[0][0]
roi_plot_md = np.where(plot_data==np.percentile(plot_data,50,interpolation='nearest'))[0][0]
roi_plot_max = np.where(plot_data==np.max(plot_data))[0][0]

roi_plot = [roi_plot_min,roi_plot_md,roi_plot_max]
    
# IDs of subjects which don't contain NaN in either of the two sessions (the sum is taken over sessions)
not_nan_id = np.setdiff1d(np.arange(ns),np.where(np.isnan(np.sum(sseg_hf[:,:,c,a,roi_plot[0]],1)))[0])

f, ax = plt.subplots(figsize=(3, 4))

dx = list( np.concatenate(( np.repeat(0,len(not_nan_id)), np.repeat(1,len(not_nan_id)), np.repeat(2,len(not_nan_id)) )) )
dy = list( np.concatenate(( trt_dice[not_nan_id,c,a,roi_plot[0]].T.flatten(), trt_dice[not_nan_id,c,a,roi_plot[1]].T.flatten(), trt_dice[not_nan_id,c,a,roi_plot[2]].T.flatten() ))  )

pt.RainCloud(x = dx, y = dy, palette = [roi_g_col_trt[2]], bw = .4, width_viol = 0.65, orient = "v", box_showfliers=False)
plt.ylim([0.4,1]); plt.yticks(fontsize=lbs-5); 
if save_fig: plt.savefig(plot_dir+'/min_md_max/dice_trt/dice_trt_nolabel.svg',bbox_inches='tight')
ax.set_xticklabels([seg[seg_match][roi_plot[0]],seg[seg_match][roi_plot[1]],seg[seg_match][roi_plot[2]]], size=lbs-2, rotation=45, ha='right'); #plt.xticks(fontsize=lbs-1.5); 
plt.ylabel('Dice coefficient', size=lbs+4); 
if save_fig: plt.savefig(plot_dir+'/min_md_max/dice_trt/dice_trt.png',bbox_inches='tight',dpi=600)

# %% Dice lohi plots

roi_plot_str = ['min','md','max']
nroi_plot = len(roi_plot_str)

# data to plot
plot_data = np.median(lohi_dice[:,c,a,:],0)

# Plot ROIs
roi_plot_min = np.where(plot_data==np.min(plot_data))[0][0]
roi_plot_md = np.where(plot_data==np.percentile(plot_data,50,interpolation='nearest'))[0][0]
roi_plot_max = np.where(plot_data==np.max(plot_data))[0][0]

roi_plot = [roi_plot_min,roi_plot_md,roi_plot_max]
    
# IDs of subjects which don't contain NaN in either of the two sessions (the sum is taken over sessions)
not_nan_id = np.setdiff1d(np.arange(ns),np.where(np.isnan(np.sum(sseg_hf[:,:,c,a,roi_plot[0]],1)))[0])

f, ax = plt.subplots(figsize=(3, 4))

dx = list( np.concatenate(( np.repeat(0,len(not_nan_id)), np.repeat(1,len(not_nan_id)), np.repeat(2,len(not_nan_id)) )) )
dy = list( np.concatenate(( lohi_dice[not_nan_id,c,a,roi_plot[0]].T.flatten(), lohi_dice[not_nan_id,c,a,roi_plot[1]].T.flatten(), lohi_dice[not_nan_id,c,a,roi_plot[2]].T.flatten() ))  )

pt.RainCloud(x = dx, y = dy, palette = [roi_g_col_lohi[1]], bw = .4, width_viol = 0.65, orient = "v", box_showfliers=False)
plt.ylim([0,1]); plt.yticks(fontsize=lbs-5); 
if save_fig: plt.savefig(plot_dir+'/min_md_max/dice_lohi/dice_lohi_nolabel.svg',bbox_inches='tight')
ax.set_xticklabels([seg[seg_match][roi_plot[0]],seg[seg_match][roi_plot[1]],seg[seg_match][roi_plot[2]]], size=lbs-2, rotation=45, ha='right'); #plt.xticks(fontsize=lbs-1.5); 
plt.ylabel('Dice coefficient', size=lbs+4); 
if save_fig: plt.savefig(plot_dir+'/min_md_max/dice_lohi/dice_lohi.png',bbox_inches='tight',dpi=600)

# %%

"""
Clinical applications
"""

# %%

if not os.path.isdir(plot_dir+'/clinical'): os.mkdir(plot_dir+'/clinical')

# %%

"""
Hippocampus
"""

c = 1   # contrast: T2w
a = -1  # acquisition: mrr

# IDs of hippocampus ROIs
hip_id = np.array([i for i in range(nroi) if 'Hippocampus' in roi[i]])

# %% ICC plots

for r in range(len(hip_id)):
    
    # ROI
    roi_nm = roi[hip_id[r]]
    
    # IDs of subjects which don't contain NaN in either of the two sessions (the sum is taken over sessions)
    not_nan_id = np.setdiff1d(np.arange(ns),np.where(np.isnan(np.sum(sseg_hf[:,:,c,a,hip_id[r]],1)))[0])

    # for fixed axis (axross both ROIs)
    xl = np.copy(sseg_hf[not_nan_id,0,c,a,:][:,hip_id]/1e3)
    yl = np.copy(sseg_hf[not_nan_id,1,c,a,:][:,hip_id]/1e3)

    # T-RT
    x = np.copy(sseg_hf[not_nan_id,0,c,a,hip_id[r]]/1e3)
    y = np.copy(sseg_hf[not_nan_id,1,c,a,hip_id[r]]/1e3)
    
    # limits and ticks
    ax_range = np.max([np.max(xl)-np.min(xl),np.max(yl)-np.min(yl)])
    ax_min = np.min([np.min(xl),np.min(yl)]) -0.1*ax_range 
    ax_max = np.max([np.max(xl),np.max(yl)]) +0.1*ax_range 
    ax_lim = [ax_min, ax_max]
    
    # plot
    plt.figure(figsize=[5,5]); plt.axis('square')                        # plot set-up
    plt.locator_params(nbins=6)
    plt.plot([ax_lim[0]-1,ax_lim[1]+1], [ax_lim[0]-1,ax_lim[1]+1], ls='--', c='gray')     # plot dashed identity line
    sb.regplot(x=x, y=y, ci=95, scatter_kws={"color": roi_g_col_trt[1]}, line_kws={"color": "black"})  # line of best fit + CI
    plt.xlim(ax_lim); plt.xticks(fontsize=axs+3)    # axis limits and ticks
    plt.ylim(ax_lim); plt.yticks(fontsize=axs+3)
    if save_fig: plt.savefig(plot_dir+'/clinical/trt_'+roi_nm.replace('.','').replace(' ','-')+'_nolabel.svg',bbox_inches='tight')
    plt.xlabel('64mT test vol. (cm$^3$)',fontsize=lbs,labelpad=10)          # labels
    plt.ylabel('64mT retest vol. (cm$^3$)',fontsize=lbs,labelpad=10)
    plt.title('ICC = '+str(round(trt_icc[c,a,hip_id[r]],2)),fontsize=lbs+2,pad=13) #+'; r = '+str(round(trt_r[-1,roi_ad_id_vol[0]],2)))
    if save_fig: plt.savefig(plot_dir+'/clinical/trt_'+roi_nm.replace('.','').replace(' ','-')+'.png',bbox_inches='tight',dpi=600)

# %% r & CCC plots

for r in range(len(hip_id)):
    
    # ROI
    roi_nm = roi[hip_id[r]]
    
    # IDs of subjects which don't contain NaN in the HFC session
    not_nan_id = np.setdiff1d(np.arange(ns),np.where(np.isnan(sseg_hf[:,0,c,a,hip_id[r]]))[0])
    
    # for fixed axis (axross both ROIs)
    xl = np.copy(sseg_hf[not_nan_id,0,c,a,:][:,hip_id]/1e3)
    yl = np.copy(sseg_ge[not_nan_id,c,:][:,hip_id]/1e3)
    
    # T-RT
    x = np.copy(sseg_hf[not_nan_id,0,c,a,hip_id[r]]/1e3)
    y = np.copy(sseg_ge[not_nan_id,c,hip_id[r]]/1e3)
    
    # limits and ticks
    ax_range = np.max([np.max(xl)-np.min(xl),np.max(yl)-np.min(yl)])
    ax_min = np.min([np.min(xl),np.min(yl)]) -0.1*ax_range 
    ax_max = np.max([np.max(xl),np.max(yl)]) +0.1*ax_range 
    ax_lim = [ax_min, ax_max]
    
    # plot
    plt.figure(figsize=[5,5]); plt.axis('square')                        # plot set-up
    plt.locator_params(nbins=6)
    plt.plot([ax_lim[0]-1,ax_lim[1]+1], [ax_lim[0]-1,ax_lim[1]+1], ls='--', c='gray')     # plot dashed identity line
    sb.regplot(x=x, y=y, ci=95, scatter_kws={"color":  roi_g_col_lohi[2]}, line_kws={"color": "black"})  # line of best fit + CI
    plt.xlim(ax_lim); plt.xticks(fontsize=axs+3)    # axis limits and ticks
    plt.ylim(ax_lim); plt.yticks(fontsize=axs+3)
    if save_fig: plt.savefig(plot_dir+'/clinical/lohi_'+roi_nm.replace('.','').replace(' ','-')+'_nolabel.svg',bbox_inches='tight')
    plt.xlabel('64mT vol. (cm$^3$)',fontsize=lbs,labelpad=10)          # labels
    plt.ylabel('3T vol. (cm$^3$)',fontsize=lbs,labelpad=10)
    # Pearson r
    pr,pp = stats.pearsonr(x,y)
    ccc = lin(x,y)
    # title and save
    plt.title('r = '+str(round(pr,2))+'; CCC = '+str(round(ccc,2)),fontsize=lbs+2,pad=13)
    if save_fig: plt.savefig(plot_dir+'/clinical/lohi_'+roi_nm.replace('.','').replace(' ','-')+'.png',bbox_inches='tight',dpi=600)

# %% Dice trt plots
    
import ptitprince as pt          # raincloud plots

# IDs of subjects which don't contain NaN in either of the two sessions (the sum is taken over sessions)
not_nan_id = np.setdiff1d(np.arange(ns),np.where(np.isnan(np.sum(sseg_hf[:,:,c,a,hip_id[0]],1)))[0])

f, ax = plt.subplots(figsize=(3, 4))

dx = list( np.concatenate(( np.repeat(0,len(not_nan_id)), np.repeat(1,len(not_nan_id)) )) )
dy = list( np.concatenate(( trt_dice[not_nan_id,c,a,hip_id[0]].T.flatten(), trt_dice[not_nan_id,c,a,hip_id[1]].T.flatten() ))  )

pt.RainCloud(x = dx, y = dy, palette = [roi_g_col_trt[2]], bw = .4, width_viol = 0.35, orient = "v", box_showfliers=False)
plt.ylim([0.65,1]); plt.yticks(fontsize=lbs-5); 
if save_fig: plt.savefig(plot_dir+'/clinical/hipp_dice_trt_nolabel.svg',bbox_inches='tight')
ax.set_xticklabels([seg[seg_match][hip_id[0]],seg[seg_match][hip_id[1]]], size=lbs-2, rotation=45, ha='right'); #plt.xticks(fontsize=lbs-1.5); 
plt.ylabel('Dice coefficient', size=lbs+4); 
if save_fig: plt.savefig(plot_dir+'/clinical/hipp_dice_trt.png',bbox_inches='tight',dpi=600)

# %% Dice lohi plots
    
# IDs of subjects which don't contain NaN in either of the two sessions (the sum is taken over sessions)
not_nan_id = np.setdiff1d(np.arange(ns),np.where(np.isnan(np.sum(sseg_hf[:,:,c,a,hip_id[0]],1)))[0])

f, ax = plt.subplots(figsize=(3, 4))

dx = list( np.concatenate(( np.repeat(0,len(not_nan_id)), np.repeat(1,len(not_nan_id)) )) )
dy = list( np.concatenate(( lohi_dice[not_nan_id,c,a,hip_id[0]].T.flatten(), lohi_dice[not_nan_id,c,a,hip_id[1]].T.flatten() ))  )

pt.RainCloud(x = dx, y = dy, palette = [roi_g_col_lohi[1]], bw = .4, width_viol = 0.35, orient = "v", box_showfliers=False)
plt.ylim([0.65,1]); plt.yticks(fontsize=lbs-5); 
if save_fig: plt.savefig(plot_dir+'/clinical/hipp_dice_lohi_nolabel.svg',bbox_inches='tight')
ax.set_xticklabels([seg[seg_match][hip_id[0]],seg[seg_match][hip_id[1]]], size=lbs-2, rotation=45, ha='right'); #plt.xticks(fontsize=lbs-1.5); 
plt.ylabel('Dice coefficient', size=lbs+4); 
if save_fig: plt.savefig(plot_dir+'/clinical/hipp_dice_lohi.png',bbox_inches='tight',dpi=600)

# %% 

"""
Deviation from the norm
"""

### residuals test

# any contrast and resolution (just for testing)
c = 0
r = 0

# before correction
test = pd.DataFrame(data={'ge_vol': sseg_ge[:,c,r], 'age': age, 'sex': sex})
g = sb.lmplot(data=test, x="age", y="ge_vol", hue="sex", height=5)
g.set_axis_labels("age (years)", "volume")

# correct
reg_res = reg_m(sseg_ge[:,c,r], [age[not_nan_id],sex_male[not_nan_id]])
test_resid = reg_res.resid

# after correction
test = pd.DataFrame(data={'ge_vol': test_resid, 'age': age, 'sex': sex})
g = sb.lmplot(data=test, x="age", y="ge_vol", hue="sex", height=5)
g.set_axis_labels("age (years)", "volume")

# %% calculate multiple regression of Hyperfine (HFC) volume as a function of age, covarying for sex

# local
sseg_hf_resid = np.zeros_like(sseg_hf)
for ss in range(nses):
    for c in range(ncon):
        for a in range(nacq):
            for r in range(nroi):
                # IDs of subjects which don't contain NaN (in the HFC / first = 0 session only, i.e. sseg_hf[:,0,c,a,r])
                not_nan_id = np.setdiff1d(np.arange(ns),np.where(np.isnan(sseg_hf[:,ss,c,a,r]))[0])
                # if not_nan_id is not empty
                if len(not_nan_id) != 0:
                    # fit model
                    reg_res = reg_m(sseg_hf[:,ss,c,a,r][not_nan_id], [age[not_nan_id],sex_male[not_nan_id]])
                    # extract residuals
                    sseg_hf_resid[:,ss,c,a,r][not_nan_id] = reg_res.resid
                # else if not_nan_id is empty (i.e. all values are NaN)
                else:
                    sseg_hf_resid[:,ss,c,a,r] = np.nan 

# %% as above, for high-field (GE) data

# local
sseg_ge_resid = np.zeros_like(sseg_ge)
for c in range(ncon):
    #print(f'--\ncontrast: {con[c]}\n--')
    for r in range(nroi):
        # fit model
        reg_res = reg_m(sseg_ge[:,c,r], [age[not_nan_id],sex_male[not_nan_id]])
        #reg_stat = reg_res.summary()
        # extract residuals
        sseg_ge_resid[:,c,r] = reg_res.resid

# %%

def non_par_z_score(vector):
    median = np.median(vector)
    mad = stats.median_abs_deviation(vector)
    
    z_scores = (vector - median) / mad
    return z_scores

# %% corrected Z-score (or Median Absolute Deviation) within each region across subjects

# hf volumes
sseg_hf_resid_z = np.zeros_like(sseg_hf)
sseg_hf_resid_zmad = np.zeros_like(sseg_hf)
for ss in range(nses):
    for c in range(ncon):
        for a in range(nacq):
            for r in range(nroi):
                not_nan_id = np.setdiff1d(np.arange(ns),np.where(np.isnan(sseg_hf[:,ss,c,a,r]))[0])
                # if not_nan_id is not empty
                if len(not_nan_id) != 0:
                    sseg_hf_resid_z[:,ss,c,a,r][not_nan_id] = stats.zscore(sseg_hf_resid[:,ss,c,a,r][not_nan_id])
                    sseg_hf_resid_zmad[:,ss,c,a,r][not_nan_id] = non_par_z_score(sseg_hf_resid[:,ss,c,a,r][not_nan_id])
                # else if not_nan_id is empty (i.e. all values are NaN)
                else:
                    sseg_hf_resid_z[:,ss,c,a,r] = np.nan
                    sseg_hf_resid_zmad[:,ss,c,a,r] = np.nan 
            
# ge volumes
sseg_ge_resid_z = np.zeros_like(sseg_ge)
sseg_ge_resid_zmad = np.zeros_like(sseg_ge)
for c in range(ncon):
    for r in range(nroi):
        sseg_ge_resid_z[:,c,r] = stats.zscore(sseg_ge_resid[:,c,r])
        sseg_ge_resid_zmad[:,c,r] = non_par_z_score(sseg_ge_resid[:,c,r])

# %% maximum region/subject in GE T2w


# GE-first
sub_max, roi_max = np.unravel_index(sseg_ge_resid_zmad[:,1,:].argmax(), sseg_ge_resid_zmad[:,1,:].shape)
np.amax(np.abs(sseg_ge_resid_zmad[:,1,:]))
sub_max
roi_max

sseg_ge_resid_zmad[sub_max,1,roi_max]
sseg_hf_resid_zmad[sub_max,0,1,4,roi_max]

# %% MAD violin plot 

# Sample data
data = {
        '3T': sseg_ge_resid_zmad[:,1,roi_max],
        '64mT test': sseg_hf_resid_zmad[:,0,1,4,roi_max],
        '64mT retest': sseg_hf_resid_zmad[:,1,1,4,roi_max]
}

# Create a DataFrame
import pandas as pd
df = pd.DataFrame(data)

# Set up the plot
plt.figure(figsize=(8, 4.5))
sb.set(style="whitegrid")

# Create the violin plots (both in grey)
ax = sb.violinplot(data=df, color="#CCCCCC", width=0.5, inner='quartiles', orient='h')

# Add individual data points on top in dark grey
sb.swarmplot(data=df, color="#000000", size=6, ax=ax, orient='h')

# Highlight one datapoint in each violin in orange
highlight_indices = [sub_max]  # Indices of the data points to highlight
highlight_color = ["#8B0000"] #["#FFA500"]
for i, group in enumerate(df.columns):
    for index in highlight_indices:
        plt.scatter(df[group].iloc[index], i, color=highlight_color, s=50, zorder=5)

# Add labels and title
ax.set_yticklabels([ '3T', '64mT$_{test}$', '64mT$_{retest}$'],fontsize=lbs)
plt.xticks(fontsize=axs);
plt.xlabel('Z$_{robust}$',fontsize=lbs+2)
plt.title(roi[roi_max],fontsize=lbs+2)

plt.tight_layout()
#plt.show()

if save_fig:
    plt.savefig(plot_dir+'/clinical/violin_sub-'+sub[sub_max]+'.svg',bbox_inches='tight')
    plt.savefig(plot_dir+'/clinical/violin_sub-'+sub[sub_max]+'.png',bbox_inches='tight',dpi=600)

# %% correlate z-values for participant sub-HYPE14

sb.set_style(style='ticks')

# HFC vs HFE
x = sseg_hf_resid_zmad[sub_max,0,1,4,:]
y = sseg_hf_resid_zmad[sub_max,1,1,4,:]

# limits and ticks
ax_lim = [-5,9] 
ax_ticks = np.arange(round(ax_min), round(ax_max)+1, 2)
# plot
plt.figure(figsize=[5,5]); plt.axis('square')                        # plot set-up
plt.plot([ax_lim[0]-1,ax_lim[1]+1], [ax_lim[0]-1,ax_lim[1]+1], ls='--', c='gray')     # plot dashed identity line
sb.regplot(x=x, y=y, ci=95, scatter_kws={"color": "grey"}, line_kws={"color": "black"})  # line of best fit + CI
plt.scatter(x[np.where(x==np.max(x))[0]],y[np.where(x==np.max(x))[0]], color="#8B0000")
plt.xlim(ax_lim); plt.xticks(ax_ticks,fontsize=axs+3)    # axis limits and ticks
plt.ylim(ax_lim); plt.yticks(ax_ticks,fontsize=axs+3)
plt.xlabel('64mT test vol. (Z$_{robust}$)',fontsize=lbs)          # labels
plt.ylabel('64mT retest vol. (Z$_{robust}$)',fontsize=lbs)
temp_icc = pg.intraclass_corr(data=pd.DataFrame({
    'subject':np.concatenate((np.arange(0,nroi),np.arange(0,nroi))),
    'scan':np.concatenate((np.repeat('T',nroi),np.repeat('RT',nroi))),
    'score':np.concatenate((x,y))}),
    targets='subject', raters='scan',ratings='score')
icc = temp_icc['ICC'][2] 
plt.title('ICC = '+str(round(icc,3)),fontsize=lbs+2) #+'; r = '+str(round(trt_r[-1,roi_ad_id_vol[0]],2)))
if save_fig:
    plt.savefig(plot_dir+'/clinical/scatter_trt_sub-'+sub[sub_max]+'.svg',bbox_inches='tight')
    plt.savefig(plot_dir+'/clinical/scatter_trt_sub-'+sub[sub_max]+'.png',bbox_inches='tight',dpi=600)

# HFC vs GE
x = sseg_hf_resid_zmad[sub_max,0,1,4,:]
y = sseg_ge_resid_zmad[sub_max,1,:]

# limits and ticks
ax_lim = [-5,9] 
ax_ticks = np.arange(round(ax_min), round(ax_max)+1, 2)
# plot
plt.figure(figsize=[5,5]); plt.axis('square')                        # plot set-up
plt.plot([ax_lim[0]-1,ax_lim[1]+1], [ax_lim[0]-1,ax_lim[1]+1], ls='--', c='gray')     # plot dashed identity line
sb.regplot(x=x, y=y, ci=95, scatter_kws={"color": "grey"}, line_kws={"color": "black"})  # line of best fit + CI
plt.scatter(x[np.where(x==np.max(x))[0]],y[np.where(x==np.max(x))[0]], color="#8B0000")
plt.xlim(ax_lim); plt.xticks(ax_ticks,fontsize=axs+3)    # axis limits and ticks
plt.ylim(ax_lim); plt.yticks(ax_ticks,fontsize=axs+3)
plt.xlabel('64mT test vol. (Z$_{robust}$)',fontsize=lbs)          # labels
plt.ylabel('3T vol. (Z$_{robust}$)',fontsize=lbs)
#r2 = r2_score(x,y)
r,p = stats.pearsonr(x,y)
l = lin(x,y)
plt.title('r = '+str(round(r,2))+'; CCC = '+str(round(l,2)),fontsize=lbs+2) #+'; r = '+str(round(trt_r[-1,roi_ad_id_vol[0]],2)))
if save_fig: 
    plt.savefig(plot_dir+'/clinical/scatter_lohi_sub-'+sub[sub_max]+'.svg',bbox_inches='tight')
    plt.savefig(plot_dir+'/clinical/scatter_lohi_sub-'+sub[sub_max]+'.png',bbox_inches='tight',dpi=600)

# %% whole-brain plots

seg_str = ['_ses-GE_T2w_synthseg', '_ses-HFC_T2w_mrr_synthseg_flirt_to_ses-GE_T2w', '_ses-HFE_T2w_mrr_synthseg_flirt_to_ses-GE_T2w']
nm_str = ['3T', '64mT test', '64mT retest']
seg_data = [sseg_ge_resid_zmad[sub_max,1,:], sseg_hf_resid_zmad[sub_max,0,1,4,:], sseg_hf_resid_zmad[sub_max,1,1,4,:]]

for n in range(3): 
    print(n)

    seg_nii = nib.load(home_dir+'/Desktop/sub-'+sub[sub_max]+'/sub-'+sub[sub_max]+seg_str[n]+'.nii.gz')
    seg_vec = np.array(seg_nii.dataobj).flatten()
    seg_mask = (seg_vec!=0)*1
    
    # extract parameters from example volume
    nvox = np.array(seg_nii.dataobj).size                                        # number of voxels
    nii_shape = np.array(seg_nii.dataobj).shape                                  # image dimensions
    nii_affine = seg_nii.affine         
    
    z_vec = np.zeros(seg_vec.shape)
    for i in range(nroi):
        z_vec[seg_vec==seg_id[seg_match][i]] = seg_data[n][i]
    
    # plot seg
    f = plt.figure(figsize=(9, 3))
    #cl = [-np.max(abs(z_vec)),np.max(abs(z_vec))]
    cl = [-deci_ceil(np.max(np.abs(seg_data)),1),deci_ceil(np.max(np.abs(seg_data)),1)]    
    plot_nl_image_masked(z_vec, seg_mask, nii_shape, nii_affine, cmap='coolwarm', clim=cl, cut_coords=np.array([-35,13,17]), draw_cross=False,black_bg=False,display_mode='ortho',figure=f,output_file=plot_dir+'/clinical/sub-'+sub[sub_max]+'_ortho_'+nm_str[n]+'_resid_zmad.png')
    plot_nl_image_masked(z_vec, seg_mask, nii_shape, nii_affine, cmap='coolwarm', clim=cl, cut_coords=np.array([-35,13,17]), draw_cross=True,black_bg=False,display_mode='ortho',figure=f,output_file=plot_dir+'/clinical/sub-'+sub[sub_max]+'_ortho_'+nm_str[n]+'_resid_zmad_cross.png')
    