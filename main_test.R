
options(repos = getOption("repos")["CRAN"])

df <- data.table::fread('Churn_Modelling.csv')
df <- as.data.frame(df)

summary(df)
str(df)
sapply(df, function(x) sum(is.na(x))) # count of missing values in each column

length(unique(df$CustomerId))

df <- na.omit(df) # remove rows with missing values
df$RowNumber <- NULL
df$CustomerId <- NULL
df$Surname <- NULL

factors_indexes <- c(2, 3, 8, 9, 11)
df[, factors_indexes] <- lapply(df[, factors_indexes], as.factor) # multiple columns to factors
sapply(df, class) # check class of each column

sapply(df, plot)

df[, -factors_indexes] <- lapply(df[,-factors_indexes], scale) # scale numeric variables to have mean = 0 and sd = 1



# logistic regression
#   feature scaling does not affect the logistic regression model, needed only when running into convergence issues
#   multicollinearity might be a problem

library(caret)

set.seed(9)
sampling_vector <- createDataPartition(df$Exited, p = 0.8, list = F)
df_train <- df[sampling_vector,]
df_test <- df[-sampling_vector,]
df_train_labels <- df$Exited[sampling_vector]
df_test_labels <- df$Exited[-sampling_vector]

logit_model <- glm(Exited ~ ., data = df_train, family = binomial('logit')) # automatically one-hot encodes factors

summary(logit_model)

#   higher z-value is better
#   AIC - lower is better model
#   null deviance - deviance of model with no features, predicting constantly 1
#   resicual deviance - model deviance
#   fishers iterations - convergence diagnostic, usually 4-8, very high if model did not converge

library(pscl)
pR2(logit_model)

#   McFadded pseudo R-squared, range 0-1, 0 - zero explained variance
#   statistical test  might be done to check if there is significant difference between null and residual deviance

train_predictions <- predict(logit_model, newdata = df_train, type = 'response')
train_class_predictions <- as.numeric(train_predictions > 0.5)
mean(train_class_predictions == df_train$Exited) # overall accuracy on train set

test_predictions <- predict(logit_model, newdata = df_test, type = 'response')
test_class_predictions <- as.numeric(test_predictions > 0.5)
mean(test_class_predictions == df_test$Exited) # overall accuracy on test set

table(as.factor(df_train$Exited), as.factor(train_class_predictions))
table(as.factor(df_test$Exited), as.factor(test_class_predictions))

library(gmodels) # for CrossTable confusion matrix
CrossTable(as.factor(df_train$Exited), as.factor(train_class_predictions))
CrossTable(as.factor(df_test$Exited), as.factor(test_class_predictions))

library(ROCR) # to plot precision-recall curve
predictions_ROCR <- prediction(train_predictions, df_train$Exited)
performance_ROCR <- performance(predictions_ROCR, measure = 'prec', x.measure = 'rec')
plot(performance_ROCR)

#   recall - % of correctly predicted class 1 from all actual belonging to class 1 (sensitivity, true positive rate)
#   precision - % of correctly predicted class 1 from all predicted of class 1 ()
#   https://www.quora.com/What-is-the-best-way-to-understand-the-terms-precision-and-recall

thresholds <- data.frame(cutoffs = performance_ROCR@alpha.values[[1]],
                        recall = performance_ROCR@x.values[[1]],
                        precision = performance_ROCR@y.values[[1]]) # get thresholds for each recall and precision

thresholds[thresholds$recall > 0.5 & thresholds$precision > 0.479,] # get thresholds for selected recall and precision

# decision trees
#   handle continuous and categorical data
#   for classification and regresison tasks
#   should handle missing values naturaly

library(C50) # allows to specify also cost matrix

cost.matrix <- matrix(c(NA, 5,
                        1, NA), 2, 2, byrow = T) # predicting 0 when true value is 1, is 5 times more costly
rownames(cost.matrix) <- colnames(cost.matrix) <- c('0', '1')

C50_model <- C5.0(Exited ~ ., data = df_train, costs = cost.matrix, control = C5.0Control(minCases = 50))
summary(C50_model)

C50_train_predictions <- predict(C50_model, df_train)
mean(df_train$Exited == C50_train_predictions)
CrossTable(df_train$Exited, C50_train_predictions)

C50_test_predictions <- predict(C50_model, df_test)
mean(df_test$Exited == C50_test_predictions)
CrossTable(df_test$Exited, C50_test_predictions)

# svm

library(e1071)

SVM_model <- svm(Exited ~., data = df_train, kernel = 'radial', cost = 1000) # tweak cost and gamma parameter

mean(df_train$Exited == SVM_model$fitted)
CrossTable(df_train$Exited, SVM_model$fitted)

SVM_test_predictions <- predict(SVM_model, df_test)
mean(df_test$Exited == SVM_test_predictions)
CrossTable(df_test$Exited, SVM_test_predictions)

set.seed(666)

SVM_model_tune <- tune(svm, Exited ~., data = df_train, kernel = 'radial',
                       ranges = list(cost = c(0.01, 0.1, 1), gamma = c(0.01, 0.5, 1)))

# xgboost

# cost sensitive classification https://mlr-org.github.io/mlr-tutorial/release/html/cost_sensitive_classif/index.html

library(xgboost) # works only with numeric data
library(dummy)
library(caret)

df_xgb <- df
df_xgb$Exited <- as.integer(as.character(df_xgb$Exited))
dfDummy <- dummy(df_xgb, int = T)
df_xgb[, sapply(df_xgb, function(x){is.ordered(x) | is.factor(x) | is.character(x)})] <- NULL
df_xgb <- cbind(df_xgb, dfDummy)

sampling_vector_xgb <- createDataPartition(df_xgb$Exited, p = 0.8, list = F)
df_train_xgb <- df_xgb[sampling_vector_xgb,]
df_test_xgb <- df_xgb[-sampling_vector_xgb,]



train_xgb_labels <- df_train_xgb$Exited
test_xgb_labels <- df_test_xgb$Exited

df_train_xgb$Exited <- NULL
df_test_xgb$Exited <- NULL



xgboost_model <- xgboost(data = data.matrix(df_train_xgb),
                         label = train_xgb_labels,
                         eta = 1,
                         nthread = -1,
                         nrounds = 10,
                         objective = 'binary:logistic',
                         max_depth = 4,
                         verbose = 2)

xgboost_train_predictions <- predict(xgboost_model, data.matrix(df_train_xgb))
xgboost_train_class_predictions <- as.numeric(xgboost_train_predictions > 0.2)
mean(train_xgb_labels == xgboost_train_class_predictions)
CrossTable(train_xgb_labels, xgboost_train_class_predictions)

xgboost_test_predictions <- predict(xgboost_model, data.matrix(df_test_xgb))
xgboost_test_class_predictions <- as.numeric(xgboost_test_predictions > 0.2)
mean(test_xgb_labels == xgboost_test_class_predictions)
CrossTable(test_xgb_labels, xgboost_test_class_predictions)


# neural network