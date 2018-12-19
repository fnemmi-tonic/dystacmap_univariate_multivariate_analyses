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
import pandas as pd

##globalCorr


data = read_csv("globalCorr_data.csv")
globalCorr_values = data.iloc[:,0:14]
subjects = data.iloc[:,14]
features_name_globalCorr = data.columns[0:14]
label = data.loc[:,"Group"]
nuisance = data.loc[:,["pre", "aix", "Sex", "Age", "MeanMotion", "MeanGlobal"]]
subj_id = np.arange(globalCorr_values.shape[0])
data = read_csv("localCorr_data.csv")
localCorr_values = data.iloc[:,0:22]
features_name_localCorr = data.columns[0:22]
data = read_csv("fALFF_data.csv")
features_name_fALFF = data.columns[0:28]
falff_values = data.iloc[:,0:28]

features_name = ["fALFF_" + str(el) for el in features_name_fALFF] + ["localCorr_" + str(el) for el in features_name_localCorr] + ["globalCorr_" + str(el) for el in features_name_globalCorr]

all_columns = np.concatenate((globalCorr_values.columns, localCorr_values.columns, falff_values.columns))

all_indexes = pd.DataFrame(np.hstack((globalCorr_values, localCorr_values, falff_values)), columns = all_columns)



#fit linear regression and take residuals
regr = LinearRegression()

all_indexes_val_nuisance_removed = np.empty(all_indexes.shape)
for n,region in enumerate(all_columns):
    regr.fit(nuisance, all_indexes.loc[:,region])
    pred = regr.predict(nuisance)
    res = all_indexes.loc[:,region] - pred
    all_indexes_val_nuisance_removed[:,n] = res
    

all_indexes_val_nuisance_removed_norm_no_td = all_indexes_val_nuisance_removed[label != 1, :]
label_no_td = label[label != 1]
subj_id_no_td = np.arange(all_indexes_val_nuisance_removed_norm_no_td.shape[0])

import pickle
with open('all_fmri_indexes_val_nuisance_removed_norm_no_td.pkl', 'wb') as f:
    pickle.dump([all_indexes_val_nuisance_removed_norm_no_td], f)




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

with open("all_fmri_results_100.pkl", "wb") as f:
    pickle.dump([real_otp_no_td, bal_no_td], f)


print(mean_confidence_interval(bal_no_td))
print(mean_confidence_interval(bal_no_td_shuff))
feat_selected = np.unique(np.concatenate(np.array(best_comb)), return_counts = True)[0][np.unique(np.concatenate(np.array(best_comb)), return_counts = True)[1].argsort()[::-1]]
time_selected = np.unique(np.concatenate(np.array(best_comb)), return_counts = True)[1][np.unique(np.concatenate(np.array(best_comb)), return_counts = True)[1].argsort()[::-1]]
feat_selected_more_than_66 = np.array(features_name)[feat_selected[time_selected > 66]]

