#Import Data

library(ISLR)
library(car)
library(dplyr)
library(ROCR)
library(pROC)
library(lubridate)
library(data.table)
library(PerformanceAnalytics)
library(xts)
library(corrplot)

vle = read.csv("Created_from_code_files/vle_full_dataset.csv")


mod1 = glm(final_result~ zero_day_ratio+stdv+total_clicks ,data=vle, family = binomial) 
vif(mod1)

vle_num = vle[,4:8]

correlations = cor(vle_num)
corrplot(correlations, diag=FALSE, type = 'lower')
