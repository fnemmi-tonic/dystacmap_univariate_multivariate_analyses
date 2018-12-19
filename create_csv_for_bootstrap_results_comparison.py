# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 10:13:51 2018

@author: federico nemmi
"""

import os
import pickle
import numpy as np
import pandas as pd

os.chdir("C:/Users/federico nemmi/Documents/Python Scripts/dystacmap")

with open("gm_results.pkl", "rb") as f:
    gm_res_true, gm_res_shuff = pickle.load(f)


with open("wm_results.pkl", "rb") as f:
    wm_res_true, wm_res_shuff = pickle.load(f)
    
with open("falff_results.pkl", "rb") as f:
    falff_res_true, falff_res_shuff = pickle.load(f)

with open("localCorr_results.pkl", "rb") as f:
    localCorr_res_true, localCorr_res_shuff = pickle.load(f)
    
with open("globalCorr_results.pkl", "rb") as f:
    globalCorr_res_true, globalCorr_res_shuff = pickle.load(f)

with open("all_struct_results.pkl", "rb") as f:
    all_struct_res_true, all_struct_res_shuff = pickle.load(f)

with open("all_fmri_results.pkl", "rb") as f:
    all_fmri_res_true, all_fmri_res_shuff = pickle.load(f)
    
with open("complete_results.pkl", "rb") as f:
    complete_res_true, complete_res_shuff = pickle.load(f)


df = pd.DataFrame(np.hstack((gm_res_true, wm_res_true, falff_res_true, 
                             localCorr_res_true, globalCorr_res_true, all_struct_res_true,
                             all_fmri_res_true, complete_res_true)), columns = ["gm", "wm", "fALFF", "localCorr", "globalCorr",
    "structural", "functional", "complete"])

df.to_csv("bootstrap_results.csv")



with open("gm_results_100_rep.pkl", "rb") as f:
    gm_res_true, gm_res_shuff = pickle.load(f)


with open("wm_results_100.pkl", "rb") as f:
    wm_res_true, wm_res_shuff = pickle.load(f)
    
with open("falff_results_100.pkl", "rb") as f:
    falff_res_true, falff_res_shuff = pickle.load(f)

with open("localCorr_results_100.pkl", "rb") as f:
    localCorr_res_true, localCorr_res_shuff = pickle.load(f)
    
with open("globalCorr_results_100.pkl", "rb") as f:
    globalCorr_res_true, globalCorr_res_shuff = pickle.load(f)

with open("all_struct_results_100.pkl", "rb") as f:
    all_struct_res_true, all_struct_res_shuff = pickle.load(f)

with open("all_fmri_results_100.pkl", "rb") as f:
    all_fmri_res_true, all_fmri_res_shuff = pickle.load(f)
    
with open("complete_results_100.pkl", "rb") as f:
    complete_res_true, complete_res_shuff = pickle.load(f)


df = pd.DataFrame(np.hstack((gm_res_true, wm_res_true, falff_res_true, 
                             localCorr_res_true, globalCorr_res_true, all_struct_res_true,
                             all_fmri_res_true, complete_res_true)), columns = ["gm", "wm", "fALFF", "localCorr", "globalCorr",
    "structural", "functional", "complete"])

df.to_csv("bootstrap_results_100_reps.csv")