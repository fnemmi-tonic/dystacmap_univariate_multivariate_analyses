# -*- coding: utf-8 -*-
"""
Created on Wed May  2 17:22:40 2018

@author: federico nemmi
"""

import os
os.chdir("C:/Users/federico nemmi/Documents/Python Scripts/dystacmap")
from pandas import read_csv
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import ttest_ind
from helper_functions import SVC_real_and_shuffled_bootstrap_feat_select
from helper_functions import mean_confidence_interval

##localCorr


data = read_csv("localCorr_data.csv")
#select only localCorr values
localCorr_values = data.iloc[:,0:22]
features_name = data.columns[0:22]
subjects = data.iloc[:,22]
label = data.loc[:,"Group"]
nuisance = data.loc[:,["pre", "aix", "Sex", "Age", "MeanMotion", "MeanGlobal"]]
subj_id = np.arange(localCorr_values.shape[0])


#fit linear regression and take residuals
regr = LinearRegression()

localCorr_val_nuisance_removed = np.empty([len(localCorr_values), len(localCorr_values.columns)])
for n,region in enumerate(localCorr_values.columns):
    regr.fit(nuisance, localCorr_values.loc[:,region])
    pred = regr.predict(nuisance)
    res = localCorr_values.loc[:,region] - pred
    localCorr_val_nuisance_removed[:,n] = res
    

localCorr_val_nuisance_removed_norm_no_td = localCorr_val_nuisance_removed[label != 1, :]
label_no_td = label[label != 1]
subj_id_no_td = np.arange(localCorr_val_nuisance_removed_norm_no_td.shape[0])

real_mean_no_td, real_std_no_td, shuffled_mean_no_td, shuffled_std_no_td, real_otp_no_td, shuffled_otp_no_td, gw_no_td, shuff_gw_no_td, bal_no_td, bal_no_td_shuff, best_comb = SVC_real_and_shuffled_bootstrap_feat_select(localCorr_val_nuisance_removed_norm_no_td, 
                                                                                                         label_no_td, subj_id_no_td, 10, n_perms = 100)

#plot_real_and_shuffled_results(real_mean_no_td, real_std_no_td, shuffled_mean_no_td, shuffled_std_no_td)
#plt.savefig("all_groups.tiff", dpi = 300)

t_test_accuracy_no_td = ttest_ind(real_otp_no_td, shuffled_otp_no_td)
t_test_bal_accuracy_no_td = ttest_ind(bal_no_td, bal_no_td_shuff)

sign_no_td = (len([num for num in shuffled_otp_no_td if num >= real_mean_no_td]) + 1)/100
sign_no_td_bal = (len([num for num in bal_no_td_shuff if num >= bal_no_td.mean()]) + 1)/100
(((shuff_gw_no_td > gw_no_td.mean(0)) * 1).sum(0))/100

import pickle

with open("localCorr_results_100.pkl", "wb") as f:
    pickle.dump([real_otp_no_td, bal_no_td], f)


print(mean_confidence_interval(bal_no_td))
print(mean_confidence_interval(bal_no_td_shuff))

feat_selected = np.unique(np.concatenate(np.array(best_comb)), return_counts = True)[0][np.unique(np.concatenate(np.array(best_comb)), return_counts = True)[1].argsort()[::-1]]
time_selected = np.unique(np.concatenate(np.array(best_comb)), return_counts = True)[1][np.unique(np.concatenate(np.array(best_comb)), return_counts = True)[1].argsort()[::-1]]
feat_selected_more_than_66 = features_name[feat_selected[time_selected > 66]]


