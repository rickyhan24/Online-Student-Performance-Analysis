###Step 1: Load and Inspect Data

library(readr)
data <- read_csv("D:/MSc - Georgia Tech/Spring 2024/Data Analytics Business - MGT-6203-OAN/Project/My Part/Total2.csv")


### Step 2: Data Preprocessing
data$gender <- as.factor(data$gender)
data$region <- as.factor(data$region)
data$highest_education <- as.factor(data$highest_education)
data$imd_band <- as.factor(data$imd_band)
data$age_band <- as.factor(data$age_band)
data$disability <- as.factor(data$disability)
data$num_missing <- as.factor(data$num_missing)
data$Cu33 <- as.factor(data$Cu33)


# Convert final_result into a binary target variable: Assuming "Pass" and "Distinction" are considered successful outcomes:
data$final_result <- ifelse(data$final_result %in% c("Pass", "Distinction"), 1, 0)
data$final_result <- as.factor(data$final_result)

# Handle missing data:
data <- na.omit(data)


### Step 3: Feature Engineering

### Step 4: Model Building

set.seed(123) # For reproducibility
library(caret)
splitIndex <- createDataPartition(data$final_result, p = .9, list = FALSE, times = 1)
trainData <- data[splitIndex,]
testData <- data[-splitIndex,]

model <- glm(final_result ~ gender + region + highest_education + imd_band + age_band + num_of_prev_attempts + studied_credits + disability + num_missing + Cu33, family = binomial(link = "logit"), data = trainData)

summary(model)


### Step 5: Model Evaluation
library(pROC)
predictions <- predict(model, testData, type = "response")
predictions_class <- ifelse(predictions > 0.5, 1, 0)
confusionMatrix(factor(predictions_class), testData$final_result)



#############
# Assuming a logistic regression model 'logit_model'
# And a test dataset 'testData'

# Load necessary library
library(pROC)

# Predict probabilities
probabilities <- predict(model, newdata = testData, type = "response")

# Compute ROC curve
roc_obj <- roc(testData$final_result, probabilities)

# Plot ROC curve
plot(roc_obj, main="ROC Curve for Logistic Regression Model")




