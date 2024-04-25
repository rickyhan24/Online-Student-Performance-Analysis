# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 21:01:13 2024

@author: brian
"""


from labellines import labelLines
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as pl
import seaborn as sns




studentassessments = pd.read_csv("OUData/studentAssessment.csv")
assessmentweights = pd.read_csv("OUData/assessments.csv")
info = pd.read_csv("OUData/studentInfo.csv")

len(studentassessments['id_student'].tolist())

###---------------------Which assessments/courses are represented in student assessments? 

no_assess_data = [each for each in assessmentweights['id_assessment'].values if each
                  not in studentassessments['id_assessment'].values]

print('Below are the asessments (all exams) that have no recorded student data')
print(assessmentweights[assessmentweights['id_assessment'].isin(no_assess_data)])

##Merge student code module and code presentation to it

keep = ['code_module', 'code_presentation', 'id_assessment']
merge_filters = ['id_assessment']


studentassessments = studentassessments.merge(assessmentweights[keep], on=merge_filters, how = 'left')


#Merge student outcome to students' assessments

keep = ['code_module', 'code_presentation', 'id_student', 'final_result']
merge_filters = ['code_module', 'code_presentation', 'id_student']

studentassessments = studentassessments.merge(info[keep], on=merge_filters, how = 'left')

info.groupby(['code_module', 'code_presentation']).size()

   ####-------Create separate df based on module code. Reset ID Student to a local id; students who are duplicates need a new id
    #####df is a dataframe of only the chosen code module with all valid presentations (BBB and DDD have excluded presentations)
    ### Change based on which course is being taken
def sort_by_module(code):    
    
    df = studentassessments[studentassessments['code_module']==code]

    if code == 'BBB':
        is_not_filter = '2014J'
        df = df[df['code_presentation'] != is_not_filter]
        
    elif code == 'DDD':
        is_not_filter = '2013B'
        df = df[df['code_presentation'] != is_not_filter]
        
    ##Save important information. Number of distinct assessments = total assessments/num of presentations
    num_pres = len(pd.unique(df['code_presentation'].values))
    num_assess = len(pd.unique(df['id_assessment'].values))
    num_assess = int(num_assess/num_pres)

    # Continue
    keep = ['id_assessment', 'id_student', 'score', 'final_result', 'code_presentation', 'code_module']
    df = df[keep]


    #Store assessments in a numpy array, note the same assessments have a common column index
    ids_assess = np.sort(pd.unique(df['id_assessment'].values)).reshape((num_pres,num_assess))


    ###Reassign all ids for each group of assessments
    for n in range(num_pres):
        
        lower = ids_assess[n][0]
        upper  = ids_assess [n][num_assess-1]
        cond1 = df['id_assessment']>= lower 
        cond2 =  df['id_assessment'] <= upper
        conditions = cond1 & cond2
        
        new_ids = df[(df['id_assessment']>= lower) & (df['id_assessment'] <= upper)]
        reassign_new_id = dict()
        for each in new_ids['id_student'].values:
            reassign_new_id[each] = each + 10**(7+n)
        
        df.loc[conditions, 'id_student'] = df.loc[conditions, 'id_student'].map(reassign_new_id)


    #####harmonize all assessment codes to the number of distinct assessments
    print(f'The module {code} has {num_assess} number of distinct assessments')


    ##Create a dictionary for mapping 
    mapping = dict()
    for each in ids_assess.flatten():
        key = each
        column = np.where(ids_assess == each)[1][0]
        replacement = ids_assess[0][column]
        mapping[key] = replacement

    df['id_assessment'] = df['id_assessment'].replace(mapping)


    ##check duplicated ids, same assessment

    duplicates = df.duplicated(subset = ['id_student', 'id_assessment'])
    assert df[duplicates].empty
    return df

#-------Create time series
def time_series (df):
    
    num_assess = int(len(pd.unique(df['id_assessment'].values)))


    pivot_columns = ['id_assessment', 'id_student', 'score']
    time_series = df[pivot_columns].pivot(index = 'id_student', columns = 'id_assessment', values = 'score' )
    
    ##How many assignments completed per group / total - NOT complete

    avg_completed = pd.DataFrame()
    avg_completed['num_missing'] = time_series.isna().sum(axis=1)
    avg_completed['num_completed'] = num_assess - time_series.isna().sum(axis=1)
    avg_completed = avg_completed.reset_index().merge(df[['final_result', 'id_student']], on = 'id_student').drop_duplicates()

    ##Fill in the na with zeros
    time_series = time_series.fillna(0)
    time_series = time_series.merge(df[['id_student', 'final_result']], on = 'id_student', how = 'left').drop_duplicates()

    return (time_series, avg_completed)

#--------------------CUSUM ----------------------------#
def cusum(time_series):
    
    means = time_series.groupby(['final_result']).mean()
    print(np.mean(means.iloc[:,1:], axis = 1))
    
    ##Finding mean and standard deviation for the control group (Pass/Distiction)
    mean_control = time_series[time_series['final_result'] == ( 'Pass' or 'Distiction' )].mean(axis=0)
    stdev_control = time_series[time_series['final_result'] == ( 'Pass' or 'Distiction' )].std(axis=0)
    
    print('The means and standard deviaitons for Distiction and Passing students are:')
    print(mean_control)
    print(stdev_control)
    
    mean_control.iloc[0]
    
    ##CUSUM to add up deviations from the mean using the formula Ci = max[ 0, !!!(-)!!!(xi - mu0) + Ci-1 - k ] 
    ### Use neagtive because we are tracking negative changes, positive do not need to be detected
    ##For distriction/pass, we anticipate xi to greater than mu0, so if it is negative, then 0 will be chosen in the max function.
    ###Ci is previous and k is a sensitivity term using standard deviation (what is a typically allowed deviation?)
    
    
    
    sample = time_series.copy().reset_index(drop=True)
    sample.set_index('id_student', inplace=True)
    sample = sample.iloc[:, :-1]
    
    CUSUM = pd.DataFrame(index=sample.index, columns=sample.columns)
    num_row, num_col = sample.shape
    
    for n,t in enumerate(sample.columns):
        
        try:
            xit = sample[t]
            mut = mean_control.iloc[n+1]
            Ci = 0 if n == 0 else CUSUM[t-1]
            k = 0*stdev_control.iloc[n+1]
        
        except:
            mut = mean_control.iloc[n+1]
            Ci = 0 if n == 0 else CUSUM[t-2]
            k = 0*stdev_control.iloc[n+1]
        
        if n == 0:
            
            diff = np.array(-(xit - mut))  + 0 - k
            diff.reshape(num_row,1)
            zeros_column = np.zeros((num_row))
            
            comparison = np.concatenate((zeros_column,diff)).reshape(2,num_row)
            St = np.max(comparison, axis = 0)
            
            CUSUM[t] = St
        
        else: 
            
            diff = np.array(-(xit - mut))  + np.array(Ci).reshape(1, num_row) - k
            zeros_column = np.zeros((1, num_row))
            
            comparison = np.concatenate((zeros_column,diff)).reshape(2,num_row)
            St = np.max(comparison, axis = 0)
            CUSUM[t] = St
            
    ### Graph of classification by predictions
    
    g = 20
    m= 10
    predictions_cu = CUSUM.iloc[g:g+m,-1] < 35
    
    actual = time_series['final_result'] == ('Distinction' or 'Pass')

    
    match = pd.DataFrame(predictions_cu.tolist()) == pd.DataFrame(actual[g:g+m].tolist())
    match.replace(True, 'g', inplace=True)
    match.replace(False, 'r', inplace = True)   
    match = match[0].to_list()
    print(match)
    
    stop = round(0.33*CUSUM.shape[1]) + 1
    CU_transpose = CUSUM.transpose().reset_index()
    
    
    x1, y1 = [0, stop-1], [35, 35]
    plt.plot(x1, y1, marker = ',')
    ind = CU_transpose.index.to_list()
    ind = ind[:stop]
    
    for n in range(m):

            
        sns.lineplot(x = ind, y = CU_transpose.iloc[ind,n+1], color = match[n])
    
    plt.ylabel('CUSUM Statistic')
    plt.xlabel('Number of Assessments Over Time')
    plt.xticks(ticks = range(stop))
    leg = plt.legend('T')
    plt.show()
    
    return CUSUM


#!!!-------------------------------------------------Use the function
code_filters = ['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG']

assess_results_all = pd.DataFrame(columns=['id_student', 'code_module', 'num_missing', 'num_completed',
                                           'Cu50', 'Cu33', 'Cu100', 'final_result',
                                           ])



                                           
                                           
for each in code_filters:
    df = sort_by_module(each)
    time, assess_results = time_series(df)
    cu_result = cusum(time)
    
    halfway = int(round(len(cu_result.columns)/2,0))
    thirdway = int(round(len(cu_result.columns)/3,0))
    
    cu_stat_last = cu_result.iloc[:,-1].rename('Cu100')
    cu_stat_half = cu_result.iloc[:, halfway].rename('Cu50')
    cu_stat_third = cu_result.iloc[:, thirdway].rename('Cu33')
        
    filtered_df = df[['id_student', 'code_module', 'code_presentation']].drop_duplicates()
    
    assess_results = assess_results.merge(cu_stat_last.reset_index(), on='id_student')
    assess_results = assess_results.merge(cu_stat_half.reset_index(), on='id_student')
    assess_results = assess_results.merge(cu_stat_third.reset_index(), on='id_student')
    assess_results = assess_results.merge(filtered_df, on='id_student', how ='inner')
    
    
    #assess_results['code_module'] = each#
    
    assess_results_all = assess_results_all.append(assess_results, ignore_index=True)

###End result has 22985 rows compared to student assessment of 23369 students 
### 26074 in student VLE data


### 1 signifies pass 0 is not pass (withdraw or fail)
bina = [1 if (x =='Pass') | (x == 'Distinction') else 0 for x in assess_results_all['final_result']]
assess_results_all['binary'] = bina

### Prediction based on thresholds (using means to determine cut offs):
print(assess_results_all.groupby(['code_module']).mean())
print(assess_results_all.groupby(['final_result']).mean())




##choose which cu stat to test (1/3, 1/2 or full course)
def  findcutoff_third():
    thresholds = range(0, 100, 5)
    predictions = assess_results_all.copy()[['binary']]
    acc = []
    recall = []
    precision = []
    specificity = []
    for t in thresholds:
        predictions['pred'] = assess_results_all['Cu33'] < t 
        
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
    print(thresholds[best_specificity])
    
    evaluation = pd.DataFrame()
    evaluation['thresholds'] = thresholds
    evaluation['accuracy'] = acc
    evaluation['recall'] = recall
    evaluation['precision'] = precision
    evaluation['specificity'] = specificity
    
    return evaluation
    

def  findcutoff_half():
    thresholds = range(60, 500, 20)
    predictions = assess_results_all.copy()[['binary']]
    acc = []
    recall = []
    precision = []
    specificity = []
    for t in thresholds:
        predictions['pred'] = assess_results_all['Cu50'] < t 
        
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
    return evaluation


def  findcutoff_whole():
    thresholds = range(60, 500, 20)
    predictions = assess_results_all.copy()[['binary']]
    acc = []
    recall = []
    precision = []
    specificity = []
    for t in thresholds:
        predictions['pred'] = assess_results_all['Cu100'] < t 
        
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
    return evaluation


eval33 = findcutoff_third()
eval50 = findcutoff_half()
eval100 = findcutoff_whole()

print(eval33)
print(eval50)
print(eval100)

###change ids back to original



def keep_rightmost_7_digits(num):
    return str(num)[-7:].lstrip('0')

digits = assess_results_all['id_student'].astype(str).apply(keep_rightmost_7_digits)

assess_results_all['id_student'] = digits.astype(int)

assess_results_all.to_csv('assessment_data_by_student.csv')

