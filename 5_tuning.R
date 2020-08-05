# Model used:
# 1. Logistic Regression
# 2. SVM
# 3. Neural Network (NNet)
# 4. Decision Tree (C5.0)
# 5. Decision Tree (AdaBoost)
# 6. Random Forest (too long)


# library(randomForest) # take too much time for training
#library(nnet)
# library(fastAdaboost)
#library(C50)
library(caret) # train(), confusionMatrix()
library(plyr) # revalue()
library(pROC) #roc()

# ----------------------------------------------------------------------------------------
# read in data
nccr.df <- read.csv("nccr_clean.csv")
dim(nccr.df) # 55724 x 28


# ----------------------------------------------------------------------------------------
# check all correlation coefficients
corr.df <- nccr.df[sample(nrow(nccr.df), 500), ]
corr.coef <- cor(corr.df)
print(corr.coef[1,]) # Presence vs. all predictors
# except above two predictors, other seems okay 
# (nlcd16_grass_pct is relatively high, need further investigation)

# remove two columns
nccr.df <- nccr.df[, !colnames(nccr.df) %in% c("BTPD_density_1km", "BTPD_density_3km")]

# cast Presence to Factor (avoid error using train())
nccr.df$Presence <- as.factor(nccr.df$Presence)
nccr.df$Presence <- revalue(nccr.df$Presence, c("1" = "Yes", "0" = "No"))
dim(nccr.df) # 55724 x 26
 
# ----------------------------------------------------------------------------------------
# 0. Data partitioning
set.seed(705968)
train.id <- sample(nrow(nccr.df), 30000)
train.df <- nccr.df[train.id,] # 30k rows
test.df <- nccr.df[-train.id,] # 25k rows

# define training parameters
cv.ctrl <- trainControl(method="repeatedcv", 
                        repeats=3,
                        summaryFunction=twoClassSummary,
                        classProbs=TRUE)


# ----------------------------------------------------------------------------------------
# 1. Logistic Regression



# tune logit regression
logit.tune <- train(Presence ~ .,
                   data=train.df,
                   method="glm",
                   metric="ROC",
                   trControl=cv.ctrl)

logit.tune # show tuning process
summary(logit.tune) # show results after tuning

pred <- predict(logit.tune, train.df)
confusionMatrix(pred, train.df$Presence) # train: 0.812

pred <- predict(logit.tune, test.df)
confusionMatrix(pred, test.df$Presence) # test: 0.811


# stepwise regression
logit.step <- step(glm(Presence ~ ., data=train.df, family=binomial("logit")),
                   direction="both")
summary(logit.step)


pred <- predict(logit.step, train.df)
pred <- ifelse(pred > 0.5, "Yes", "No")
mean(pred == train.df$Presence) # train: 0.814

pred <- predict(logit.step, test.df)
pred <- ifelse(pred > 0.5, "Yes", "No")
mean(pred == test.df$Presence) # train: 0.815
# slightly improvements


# ROC curve
prob <- predict(logit.tune, test.df, type="prob")
ROC <- roc(response=test.df$Presence,
           predictor=prob$Yes,
           levels=levels(test.df$Presence))
plot(ROC, type="S")


# ----------------------------------------------------------------------------------------
# 2. SVM
library(kernlab)

# parallel processing
library(doParallel) 
cl <- makePSOCKcluster(8)
registerDoParallel(cl)

# linear kernel
svm.linear.Grid <- expand.grid(C=seq(1, 10, 1))
svm.linear.tune <- train(Presence~.,
                         data=train.df,
                         method="svmLinear2",
                         metric="ROC",
                         trControl=cv.ctrl,
                         tuneGrid=expand.grid(cost=1))


library(e1071)
svm.model <- svm(Presence~., data=train.df, scale=TRUE, kernel="polynomial", cost=10)

svm.pred <- predict(svm.model, newdata=train.df)
mean(svm.pred == train.df[, 1]) * 100 # train: 90.87%

svm.pred <- predict(svm.model, newdata=test.df)
mean(svm.pred == test.df[, 1]) * 100 # test: 89.06%



# polynomial 



svmGrid <- expand.grid(mtry=seq(1, 10, 1))
svm.tune <- train(Presence ~ .,
                 data=train.df,
                 method="rf",
                 metric="ROC",
                 trControl=cv.ctrl)
summary(rf.tune)
















# ----------------------------------------------------------------------------------------
# 2. Random Forest

# use parallel processing package to speed-up 
cl <- makePSOCKcluster(8)
registerDoParallel(cl)

rfGrid <- expand.grid(mtry=seq(1, 10, 1))
rf.tune <- train(Presence ~ .,
                 data=train.df,
                 method="rf",
                 metric="ROC",
                 trControl=cv.ctrl)
summary(rf.tune)

  
# ----------------------------------------------------------------------------------------
# 3. Neural Network


  
  
  
  
  
