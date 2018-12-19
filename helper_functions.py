# -*- coding: utf-8 -*-
"""
Created on Wed May  2 16:25:16 2018

@author: federico nemmi
"""
import scipy as sp
import scipy.stats
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from scipy.stats import itemfreq
from sklearn.utils import resample
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC



def SVC_real_and_shuffled_bootstrap_feat_select(features, labels, subj_id,  r_seed, n_perms = 5000):
    #crea tutti gli array vuoit che verranno popolati nella funzione
    otp_real = np.empty([n_perms, 1])
    otp_shuffled = np.empty([n_perms, 1])
    group_wise_accuracy = np.empty([n_perms, len(labels.unique())])
    group_wise_accuracy_shuffled = np.empty([n_perms, len(labels.unique())])
    balanced_accuracy = np.empty([n_perms, 1])
    balanced_accuracy_shuffled = np.empty([n_perms, 1])
    feat_index = np.arange(features.shape[1])
    selected_features = []
    
    #loop over il numero di bootstrapping
    for n in range(0,n_perms):
        print ("Iteration advancement = {0:3.2f}%".format((n/n_perms)*100))
        #il while loop è qui per accertarsi che ci sia almeno un esemplare di ciascuna categoria nel resampled sample
        test = 0
        while test == 0:
            res_features, res_labels, res_subj = resample(features, labels, subj_id, random_state = r_seed + n)
            test = all(itemfreq(res_labels)[:,1] > 0) * 1
        #dividi in train and testing sets
        X_train = res_features
        y_train = res_labels
        oob_subj = [s for s in subj_id if s not in res_subj]
        X_test = np.array([features[n,:] for n in np.arange(features.shape[0]) if n in oob_subj])
        y_test = [labels.iloc[n] for n in np.arange(features.shape[0]) if n in oob_subj]
        
        #fitta il navie gaussian e predici
        clf = LinearSVC()
        clf.fit(X_train, y_train)
        model = SelectFromModel(clf, prefit = True)
        selected_features.append(feat_index[model.get_support()])
        clf_fin = LinearSVC()
        clf_fin.fit(model.transform(X_train), y_train)
        pred = clf_fin.predict(model.transform(X_test))
        #calcola l accuracy
        otp_real[n] = accuracy_score(y_test, pred)
        #calcola l accuracy per ciascuna classe
        conf = confusion_matrix(y_test, pred)
        temp_conf = np.diag(conf)/conf.sum(1)
        group_wise_accuracy[n,:] = temp_conf.transpose()
        #calcola la balanced accuracy
        balanced_accuracy[n] = temp_conf.mean()
    
    

    
        X_train = shuffle(X_train, random_state = r_seed + n)
        clf = LinearSVC()
        clf.fit(X_train, y_train)
        model = SelectFromModel(clf, prefit = True)
        clf_fin = LinearSVC()
        clf_fin.fit(model.transform(X_train), y_train)
        pred = clf_fin.predict(model.transform(X_test))
        otp_shuffled[n] = accuracy_score(y_test, pred)
        conf = confusion_matrix(y_test, pred)
        temp_conf_shuffled = np.diag(conf)/conf.sum(1)
        group_wise_accuracy_shuffled[n,:] = temp_conf_shuffled.transpose()
        balanced_accuracy_shuffled[n] = temp_conf_shuffled.mean()
        
    otp_mean = otp_real.mean()
    otp_std = otp_real.std()    
    otp_shuffled_mean = otp_shuffled.mean()
    otp_shuffled_std = otp_shuffled.std()
    
    
    
    return otp_mean, otp_std, otp_shuffled_mean, otp_shuffled_std, otp_real, otp_shuffled, group_wise_accuracy, group_wise_accuracy_shuffled, balanced_accuracy, balanced_accuracy_shuffled, selected_features

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h


