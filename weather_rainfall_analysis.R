#Loading necessary libraries
library(car)
library(MASS)
library(leaps)
library(ggplot2)
library(dplyr)
library(glmnet)

#Loading Data
data = read.csv("data/weatherAUS.csv", sep = ",")

#Summarizing the data
data
head(data)
str(data)
summary(data)
summary(data$Rainfall)
#In the "Rainfall" variable, zeros dominate, and the median is 0

#Creating a smaller subset for visualization (full dataset was slow)
set.seed(123)
random_index = sample(1:nrow(data), size = 1000)
data_small = data[random_index, ]
visualisation = c("Rainfall", "MinTemp", "MaxTemp", "Humidity9am", "Pressure9am", "Cloud9am", "WindSpeed9am")

#Creating plots for selected pairs of variables
pairs(data_small[, visualisation])

#Manually selecting potential predictors
predictors = c("Rainfall", "MinTemp", "MaxTemp", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm", "WindSpeed9am", "WindSpeed3pm")
df_clean = data %>%
  dplyr::select(all_of(predictors)) %>%
  na.omit()
dim(df_clean)

#Splitting into training and test sets
set.seed(123)
n = nrow(df_clean)
train_index = sample(1:n, size = round(0.7 * n)) 
train = df_clean[train_index, ]
test  = df_clean[-train_index, ]

#Log-transforming the "Rainfall" variable
train$log_Rainfall = log(train$Rainfall + 0.1)

#Comparing the distribution of "Rainfall" before and after log transformation
par(mfrow = c(1, 2))
plot(density(train$Rainfall))
plot(density(train$log_Rainfall))
par(mfrow = c(1, 1))
#The distribution on the left is heavily skewed with many zeros
#The distribution on the right has reduced skewness but is still bimodal

#Building the full linear model
full_model = lm(log_Rainfall ~ ., data = train %>% dplyr::select(-Rainfall))
summary(full_model)

#Creating a smaller subset of data for model visualization
random_index = sample(1:length(fitted(full_model)), size = 1000)
residual_sample = rstandard(full_model)[random_index]
fitted_sample = fitted(full_model)[random_index]
sqrt_abs_rest_sample = sqrt(abs(rstandard(full_model)))[random_index]
leverage_sample = hatvalues(full_model)[random_index] 
par(mfrow = c(2, 2)) 

#Plot 1: Residuals vs Fitted (Linearity and Zero Mean)
plot(fitted_sample, 
     residual_sample,
     main = "Residuals vs Fitted (N=1000)",
     xlab = "Fitted Values (log)",
     ylab = "Standardized Residuals",
     pch = 19, cex = 0.5)
abline(h = 0, lty = 2, col = "gray")

#Plot 2: Normal Q-Q Plot (Residual Normality)
qqnorm(residual_sample,
       main = "Normal Q-Q Plot (N=1000)",
       xlab = "Theoretical Quantiles",
       ylab = "Standardized Residuals",
       pch = 19, cex = 0.5)
qqline(residual_sample, col = "red") 

#Plot 3: Scale-Location (Homoscedasticity / Constant Variance)
plot(fitted_sample,
     sqrt_abs_rest_sample,
     main = "Scale-Location (N=1000)",
     xlab = "Fitted Values (log)",
     ylab = expression(sqrt(abs("Standardized Residuals"))),
     pch = 19, cex = 0.5)

#Plot 4: Residuals vs Leverage (Influential Observations)
plot(leverage_sample,
     residual_sample,
     main = "Residuals vs Leverage (N=1000)",
     xlab = "Leverage",
     ylab = "Standardized Residuals",
     pch = 19, cex = 0.5)

par(mfrow = c(1, 1)) 
#Plot 1: Residuals show pattern, suggesting possible model misfit
#Plot 2: Deviations from the line, especially at the tails
#Plot 3: Heteroskedasticity (residual variance is not constant)
#Plot 4: No obviously extreme influential observations

#Checking multicollinearity (VIF)
vif(full_model)

#Both stepAIC and Best Subset Selection indicate the full model is optimal according to information criteria.
step_model <- stepAIC(full_model, direction = "both", trace = TRUE)
summary(step_model)
formula(step_model)
final_model = step_model

#Preparing model matrices for LASSO
x_train <- model.matrix(log_Rainfall ~ MinTemp + MaxTemp + Humidity9am + Humidity3pm + 
                          Pressure9am + Pressure3pm + Cloud9am + Cloud3pm + 
                          WindSpeed9am + WindSpeed3pm, 
                        data = train)
x_train <- x_train[, -1]
y_train <- train$log_Rainfall

#Fitting LASSO regression
set.seed(123)
cvfit <- cv.glmnet(x_train, y_train, 
                   alpha = 1,
                   nfolds = 10)

#Selecting optimal lambda values
lambda_min <- cvfit$lambda.min
lambda_1se <- cvfit$lambda.1se

#Checking model predictions on the test set
pred_log = predict(step_model, newdata = test)

#Predictions are made using the model selected by stepAIC (final_model)

#Plot 1: Regularization path (Penalty Path Plot)
plot(cvfit$glmnet.fit, xvar = "lambda", label = TRUE)

#Plot 2: Cross-validation plot (CV Plot)
plot(cvfit)
abline(v = log(lambda_min), lty = 2, col = "red")  #lambda.min
abline(v = log(lambda_1se), lty = 2, col = "blue") #lambda.1se

#Both stepAIC and Best Subset Selection indicate that the full model is optimal according to information criteria
#Coefficients for lambda.min
coef_min <- coef(cvfit, s = "lambda.min")
print("Coefficients for lambda.min:")
print(coef_min)

#Coefficients for the more conservative lambda
coef_1se <- coef(cvfit, s = "lambda.1se")
print("Coefficients for lambda.1se:")
print(coef_1se)

#LASSO suggests removing one variable for the more conservative lambda (lambda.1se), but AIC, BIC, and Best Subset Selection indicate the full model is optimal, therefore all predictors are kept in the final analysis.

#1. Preparing the dataframe for regsubsets (only Y and X variables). Only columns including "log_Rainfall" are used.
df_regsubsets <- train %>% 
  dplyr::select(-Rainfall)

#2. Running Best Subset Selection. nbest = 1: select only the single best model for each number of predictors.
best_subset <- regsubsets(log_Rainfall ~ ., 
                          data = df_regsubsets, 
                          nbest = 1) 

#3. Generating grid plots. Using adjusted R-squared (adjr2) or BIC (Bayesian Information Criterion).

par(mfrow = c(1, 2))
#Plot 1: Variable selection based on adjusted R-squared
plot(best_subset, 
     scale = "adjr2", 
     main = "Variable selection: adjusted R-squared")

#Plot 2: Variable selection based on BIC
plot(best_subset, 
     scale = "bic", 
     main = "Variable selection: BIC")

#Although Best Subset Selection suggests dropping two variables, and LASSO suggests slight reduction for a conservative lambda, we keep all predictors in the final model to match stepAIC and for consistency in the analysis.

par(mfrow = c(1, 1))

#Reversing the logarithmic transformation
pred_rainfall_calc = exp(pred_log) - 0.1
pred_rainfall_final = pmax(0, pred_rainfall_calc)

#Calculating RMSE
actual_rainfall = test$Rainfall
rmse = function(y, yhat) sqrt(mean((y - yhat)^2))
rmse_final = rmse(actual_rainfall, pred_rainfall_final)
print(paste("Final RMSE on the test set:", rmse_final, "mm"))

#Average Rainfall value
mean(train$Rainfall)

