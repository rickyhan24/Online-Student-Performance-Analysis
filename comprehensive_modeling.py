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




all_data = pd.read_csv("Created_from_code_files/comprehensive_all.csv")
demo_results = pd.read_csv("Created_from_code_files/predicted_demo_logistic.csv")
interaction_results = pd.read_csv("Created_from_code_files/predicted_comp_interaction.csv")
comp_results = pd.read_csv("Created_from_code_files/predicted_comp_logistic.csv")
step_results = pd.read_csv("Created_from_code_files/predicted_comp_logistic_step.csv")


all_data['demo_pred'] = demo_results['x']
all_data['comp_pred'] = comp_results['x']
all_data['comp_pred_step'] = step_results['x']
all_data['interaction'] = interaction_results['x']


def  findbest_model():
    columnss = ['demo_pred', 'interaction', 'comp_pred', 'comp_pred_step']
    for col in columnss:
        predictions = all_data.copy()[['binary']]
        thresholds = range(0, 100, 5)
        acc = []
        recall = []
        precision = []
        specificity = []
        for t in thresholds:
            predictions['pred'] = all_data[col] > (t/100)
            
            acc.append(accuracy_score(predictions['binary'], predictions['pred']))
            recall.append(recall_score(predictions['binary'], predictions['pred'], pos_label=0))
            precision.append(precision_score(predictions['binary'], predictions['pred'], pos_label=0))
            specificity.append(recall_score(predictions['binary'], predictions['pred'], pos_label=1))
        
        best_acc = np.argmax(acc)
        best_recall = np.argmax(recall)
        best_precision = np.argmax(precision)
        best_specificity = np.argmax(specificity)
        
        print(thresholds[best_acc])
        print(thresholds[best_recall])
        print(thresholds[best_precision])
        
        evaluation = pd.DataFrame()
        evaluation['thresholds'] = thresholds
        evaluation['accuracy'] = acc
        evaluation['recall'] = recall
        evaluation['precision'] = precision
        evaluation['specificity'] = specificity
        print(evaluation)


findbest_model()


