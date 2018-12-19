# -*- coding: utf-8 -*-
"""
Created on Wed May  2 17:22:40 2018

@author: federico nemmi
"""

import os
os.chdir("C:/Users/federico nemmi/Documents/Python Scripts/dystacmap")
from pandas import read_csv
import numpy as np
from scipy.stats import ttest_ind
from helper_functions import SVC_real_and_shuffled_bootstrap_feat_select
from helper_functions import mean_confidence_interval
import pickle

with open("gm_val_nuisance_removed_norm_no_td.pkl", "rb") as f:  
    gm_val_nuisance_removed_norm_no_td = pickle.load(f)[0]


with open("wm_val_nuisance_removed_norm_no_td.pkl", "rb") as f:  
    wm_val_nuisance_removed_norm_no_td = pickle.load(f)[0]

data = read_csv("gm_data.csv")
subjects = data.iloc[:,16]
features_name_gm = data.columns[0:16]
data = read_csv("wm_data.csv")
features_name_wm = data.columns[0:6]
features_name = ["gm_" + str(el) for el in features_name_gm] + ["wm_" + str(el) for el in features_name_wm]
label = data.loc[:,"Group"]
subj_id = np.arange(data.values.shape[0])
all_indexes_val_nuisance_removed_norm_no_td = np.hstack((gm_val_nuisance_removed_norm_no_td, wm_val_nuisance_removed_norm_no_td))
label_no_td = label[label != 1]
subj_id_no_td = np.arange(all_indexes_val_nuisance_removed_norm_no_td.shape[0])



real_mean_no_td, real_std_no_td, shuffled_mean_no_td, shuffled_std_no_td, real_otp_no_td, shuffled_otp_no_td, gw_no_td, shuff_gw_no_td, bal_no_td, bal_no_td_shuff, best_comb = SVC_real_and_shuffled_bootstrap_feat_select(all_indexes_val_nuisance_removed_norm_no_td, 
                                                                                                         label_no_td, subj_id_no_td, 10, n_perms = 100)

#plot_real_and_shuffled_results(real_mean_no_td, real_std_no_td, shuffled_mean_no_td, shuffled_std_no_td)
#plt.savefig("all_groups.tiff", dpi = 300)

t_test_accuracy_no_td = ttest_ind(real_otp_no_td, shuffled_otp_no_td)
t_test_bal_accuracy_no_td = ttest_ind(bal_no_td, bal_no_td_shuff)

sign_no_td = (len([num for num in shuffled_otp_no_td if num >= real_mean_no_td]) + 1)/100
sign_no_td_bal = (len([num for num in bal_no_td_shuff if num >= bal_no_td.mean()]) + 1)/100
(((shuff_gw_no_td > gw_no_td.mean(0)) * 1).sum(0))/100

import pickle

with open("all_struct_results_100.pkl", "wb") as f:
    pickle.dump([real_otp_no_td, bal_no_td], f)

print(mean_confidence_interval(bal_no_td))
print(mean_confidence_interval(bal_no_td_shuff))
feat_selected = np.unique(np.concatenate(np.array(best_comb)), return_counts = True)[0][np.unique(np.concatenate(np.array(best_comb)), return_counts = True)[1].argsort()[::-1]]
time_selected = np.unique(np.concatenate(np.array(best_comb)), return_counts = True)[1][np.unique(np.concatenate(np.array(best_comb)), return_counts = True)[1].argsort()[::-1]]
feat_selected_more_than_66 = np.array(features_name)[feat_selected[time_selected > 66]]


