
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
library(caret)



dem_data = read.csv("Created_from_code_files/demographic_all.csv")
comp_data = read.csv("Created_from_code_files/comprehensive_all.csv")

dem_data = dem_data[,-1]
comp_data = comp_data[,-1]



mod0 = glm(binary ~., data = dem_data, family = binomial)
summary(mod0)


mod1 = glm(binary ~ ., data = comp_data, family = binomial(link = 'logit'))
summary(mod1)

predicted_null = predict(mod0, dem_data, type = 'response')
predicted_comp = predict(mod1, comp_data, type = 'response')

mod2 = glm(formula = binary ~ total_clicks + code_module + highest_education + 
             zero_day_ratio + imd_band + studied_credits, family = binomial(link = "logit"), 
           data = comp_data)

predicted_interaction = predict(mod2, comp_data, type = 'response')


min.model= glm(binary ~ Cu33+ total_clicks, data = comp_data, family = binomial(link = 'logit'))
biggest <- formula(glm(binary~., data=comp_data, family=binomial))

step(min.model, direction = 'both', scope = biggest)

mod3 = glm(formula = binary ~ Cu33 + total_clicks + code_module + highest_education + 
             zero_day_ratio + imd_band + studied_credits, family = binomial(link = "logit"), 
           data = comp_data)

predicted_comp_step = predict(mod3, comp_data, type = 'response')


write.csv(predicted_null, '~/GA Tech/MGT 6203/Project/predicted_demo_logistic.csv')
write.csv(predicted_comp, '~/GA Tech/MGT 6203/Project/predicted_comp_logistic.csv')
write.csv(predicted_interaction, '~/GA Tech/MGT 6203/Project/predicted_comp_interaction.csv')
write.csv(predicted_comp_step, '~/GA Tech/MGT 6203/Project/predicted_comp_logistic_step.csv')
