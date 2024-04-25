# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 19:33:24 2024

@author: brian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import sklearn as sk




assessments = pd.read_csv("Created_from_code_files/merged_info_assessments.csv")
vle = pd.read_csv("Created_from_code_files/vle_full_dataset.csv")

print(assessments.columns)

all_data = assessments.merge(vle, how = 'inner', on = ['id_student', 'code_module', 'code_presentation'])

all_data['imd_band'].replace("10-20", "10-20%" , inplace=True)
all_data['imd_band'].replace('', np.nan, inplace=True)

all_data.dropna(inplace=True)

keep_filter = ['code_module',
               'gender', 'highest_education', 'imd_band', 'age_band',
               'studied_credits', 'disability',
               'Cu33', 'binary',
               'zero_day_ratio', 'stdv', 'total_clicks']

 

demo_filter = ['code_module',
               'gender', 'highest_education', 'imd_band', 'age_band',
               'studied_credits', 'disability',
               'binary']
               

comprehensive_data = all_data[keep_filter]
demographic_data = all_data[demo_filter]

demographic_data.to_csv('demographic_all.csv')
comprehensive_data.to_csv('comprehensive_all.csv')
