# dystacmap_univariate_multivariate_analyses
This repository contains the functions and scripts that have been used for an ongoing study of children with dyslexia, dyspraxia and comorbidity between the two.
The aim was to find the best modalities/combination of modalities that could discriminate between the groups using as features the results of univariate analyses (performed in SPM) comparing
these 3 groups to healthy controls.
For this study I have used SVC and bootstrap methods.

helper_functions.r contains all companion functions nedded to run the analyses
SVC* contain the script for classifying children using single/multiple modalities
create_csv_for_bootstrap_results_comparison.py stich together all results for further analysis in R
comparison_of_calissifer_stats_and_plots.r statistically compare the performance of the different classifiers and output the relevant plot
