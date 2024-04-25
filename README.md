# Team-111
  
 Team 111		Brian McClure, Richard Han, Maysam Molana, John Redden 
 
 **Factors Influencing Success in Online College**
 
Base OULAD Dataset: Dataset: https://analyse.kmi.open.ac.uk/open_dataset

**Dropbox Link Custom Datasets:** https://www.dropbox.com/scl/fo/wz291mspowj8pglt9ciu0/AIOUF-OJqS0hJ5EkaiIwjvs?rlkey=7kg8ervqsdp8ao4kxvi94e83z&st=dq67hp4r&dl=0


*** *** ***

**EDA and Study 1** Basic EDA, Charts, and IMD Poverty Study (R Markdown Files):

Initial graphs from studentInfo.csv in OULAD dataset:  **Basic-EDA.Rmd**

IMD Poverty Band Study (R Markdown File): **IMD-Poverty-Band.Rmd**

3 Classification Models (Demographics): **Basic-Models.Rmd**

3 Classification Models (Demographics + VLE): **Merged-StudentInfo-VLE.Rmd**

3 Classification Models (Demographics + VLE + Cu33): **Comprehensive-Model.Rmd**

*** *** ***

**Study 2:** VLE study (time series)

The clickstream data are plotted for some sample students from different types of student--passing, withdrawn, failing: **VLE EDA.ipynb**  

The mean daily clicks are plotted over time for each type of student in aggregate for a sample module and presentation: **VLE EDA Part 2.ipynb**  

The mean daily clicks are plotted over time for each type of student in aggregate for all modules and presentations.  The 7-day moving averages are also plotted for each type of student. **VLE EDA Part 3.ipynb**  

Some predictive modelling is performed for a sample module and presentation using student interaction features. **VLE Prediction.ipynb**  

A dataset that contains student interaction features for each student is created to be handed off to the comprehensive model. **VLE Dataset Creation.ipynb**  

Investigates the correlation and VIF between VLE interaction variables **vle_exploration**

*** *** ***

**Study 3:** VLE study (time series)

**CUMSUM student assessments.py** -- Import original data from OULAData folder; sorts and filters studentinfo dataset and studentassessment data by code module. Data is cleaned, exluding one course presentation from code module BBB and DDD each. A time series is created for each student in each module. Imputed values for NAN are 0 score for each NAN assessment. CUSUM statistic calculated using mean scores and standard deviation for each assessment in each module for 3 different cutoffs (33%, 50%, and 100% of the course assessments available). Assessment_all has CUSUM statistics at each of the three checkpoints. The most important is 33%. Code also classifies pass/fail across different thresholds and the resulting sensitivity, accuracy, precision, specificity.

Analysis of classification metrics by course module. **explore AAA, BBB, CCC, etc**

*** *** ***

**Study 4:** Threshold Tuning for Logistic Regression

Combines all 3 data sources (demographics, VLE, assessment) into one: **combining data export.py**

Uses **comprehensive_all.csv** to create logistic regression models for varying feature combinations. Outputs 4 different prediction csv's **logistic regression export prediction.R**

Calculates metrics for classification (sensitivity, precision, specificity, accuracy) using the predictions from previous R file **comprehensive_modeling**


