# Modeling
library(caret) # train(), confusionMatrix()
library(plyr) # revalue()
library(pROC) #roc()

# 1. Logistic
# ----------------------------------------------------------------------------------------
train.df <- read.csv("fb_pca_train.csv")
train.df$Presence <- as.factor(train.df$Presence)
train.df$Presence <- revalue(train.df$Presence, c("1" = "Yes", "0" = "No"))

test.df <- read.csv("fb_pca_test.csv")
test.df$Presence <- as.factor(test.df$Presence)
test.df$Presence <- revalue(test.df$Presence, c("1" = "Yes", "0" = "No"))

cv.ctrl <- trainControl(method="cv", 
                        number=5,
                        summaryFunction=twoClassSummary,
                        classProbs=TRUE,
                        verboseIter=TRUE)

# tune logit regression
set.seed(705968)
logit.tune <- train(Presence~.,
                   data=train.df,
                   method="glm",
                   metric="ROC",
                   trControl=cv.ctrl)

logit.tune # show tuning process
summary(logit.tune) # show results after tuning

pred <- predict(logit.tune, train.df)
confusionMatrix(pred, train.df$Presence) # train: 0.7451 (PCA_15)

pred <- predict(logit.tune, test.df)
confusionMatrix(pred, test.df$Presence) # test: 0.7453 (PCA_15)



# ROC curve
prob <- predict(logit.tune, test.df, type="prob")
ROC <- roc(response=test.df$Presence,
           predictor=prob$Yes,
           levels=levels(test.df$Presence))
plot(ROC, type="S")



# 2. SVM 
# ----------------------------------------------------------------------------------------
# parallel computing
library(doParallel) 
cl <- makePSOCKcluster(8, outfile="")
registerDoParallel(cl)

library(e1071)
set.seed(705968)
svm.model <- svm(Presence~., data=train.df, scale=TRUE, kernel="polynomial", cost=10)

svm.pred <- predict(svm.model, newdata=train.df)
mean(svm.pred == train.df[, 1]) * 100 # train: 85.75% (16)

svm.pred <- predict(svm.model, newdata=test.df)
mean(svm.pred == test.df[, 1]) * 100 # test: 82.74% (16)




# SVM polynomial kernel (using kernlab)
library(kernlab)
# degree = 10
# scale = 0.01
# C = 1
svm.poly.tune <- train(Presence~.,
                       data=train.df,
                       method="svmPoly",
                       metric="ROC",
                       trControl=cv.ctrl,
                       tuneGrid=expand.grid(degree=c(3, 5, 10),
                                            scale=c(1, 0.5, 0.01),
                                            C=c(0.5, 1)))

svm.poly.tune # show tuning process
summary(svm.poly.tune) # show results after tuning

pred <- predict(svm.poly.tune, train.df)
confusionMatrix(pred, train.df$Presence) # train: 0.8729

pred <- predict(svm.poly.tune, test.df)
confusionMatrix(pred, test.df$Presence) # test: 0.8489



# ROC curve
prob <- predict(svm.poly.tune, test.df, type="prob")
ROC <- roc(response=test.df$Presence,
           predictor=prob$Yes,
           levels=levels(test.df$Presence))

plot(ROC, type="S")


# 3. naive Bayesian
# ----------------------------------------------------------------------------------------
library(naivebayes)

set.seed(705968)
bayes.model <- naiveBayes(Presence~., data=train.df)

bayes.pred <- predict(bayes.model, newdata=train.df)
mean(bayes.pred == train.df[, 1]) * 100 # train: 70.14%

bayes.pred <- predict(bayes.model, newdata=test.df)
mean(bayes.pred == test.df[, 1]) * 100 # test: 69.97%


# tuning (parameters do not affect any results)
bayes.tune <- train(Presence~.,
                    data=train.df,
                    method="naive_bayes",
                    metric="ROC",
                    trControl=cv.ctrl)

bayes.tune # show tuning process
summary(bayes.tune) # show results after tuning

pred <- predict(bayes.tune, train.df)
confusionMatrix(pred, train.df$Presence) # train: 0.7647

pred <- predict(bayes.tune, test.df)
confusionMatrix(pred, test.df$Presence) # test: 0.7469



# ROC curve
prob <- predict(bayes.tune, test.df, type="prob")
ROC <- roc(response=test.df$Presence,
           predictor=prob$Yes,
           levels=levels(test.df$Presence))
plot(ROC, type="S")



# 4. Decision Tree
# ----------------------------------------------------------------------------------------
# install.packages("rpart")
library(rpart)
tree.model <- rpart(Presence~., data=train.df, method="class")
summary(tree.model) # can visualize the tree

tree.pred <- predict(tree.model, newdata=train.df, type="class")
mean(tree.pred == train.df[, 1]) * 100 # train: 75.01%

tree.pred <- predict(tree.model, newdata=test.df, type="class")
mean(tree.pred == test.df[, 1]) * 100 # test: 74.03%


# tuning (parameters do not affect any results)
# using "rpart2" using "maxdepth" instead of "rpart" using "cp (complexity parameter)"
tree.tune <- train(Presence~.,
                   data=train.df,
                   method="rpart2",
                   metric="ROC",
                   trControl=cv.ctrl)

tree.tune # show tuning process
#summary(tree.tune) # show results after tuning

pred <- predict(tree.tune, train.df)
confusionMatrix(pred, train.df$Presence) # train: 0.7501

pred <- predict(tree.tune, test.df)
confusionMatrix(pred, test.df$Presence) # test: 0.7403


# ROC curve
prob <- predict(tree.tune, test.df, type="prob")
ROC <- roc(response=test.df$Presence,
           predictor=prob$Yes,
           levels=levels(test.df$Presence))
plot(ROC, type="S")



# visualize the tree
library(rpart.plot)
rpart.plot(tree.model, type=4, faclen=0)



# 5. NNet (single hidden layer Neural Net)
# ----------------------------------------------------------------------------------------
library(nnet)

nnet.model <- nnet(Presence~., data=train.df, method="class", size=5)
summary(tree.model)

nnet.pred <- predict(nnet.model, newdata=train.df, type="class")
mean(nnet.pred == train.df[, 1]) * 100 # train: 80.75%

nnet.pred <- predict(nnet.model, newdata=test.df, type="class")
mean(nnet.pred == test.df[, 1]) * 100 # test: 79.53%


# tuning 
# best model: size=50, decay=0.5
nnet.tune <- train(Presence~.,
                   data=train.df,
                   method="nnet",
                   metric="ROC",
                   trControl=cv.ctrl,
                   tuneGrid=expand.grid(size=c(40, 50, 55),
                                        decay=c(0, 0.5, 0.01, 0,001)))

nnet.tune # show tuning process
#summary(nnet.tune) # show results after tuning

pred <- predict(nnet.tune, train.df)
confusionMatrix(pred, train.df$Presence) # train: 0.8991

pred <- predict(nnet.tune, test.df)
confusionMatrix(pred, test.df$Presence) # test: 0.8580


# ROC curve
prob <- predict(nnet.tune, test.df, type="prob")
ROC <- roc(response=test.df$Presence,
           predictor=prob$Yes,
           levels=levels(test.df$Presence))
plot(ROC, type="S")




# 6. Random Forest
# ----------------------------------------------------------------------------------------
library(randomForest)


rf.model <- randomForest(Presence~., data=train.df)
summary(rf.model)

rf.pred <- predict(rf.model, newdata=train.df, type="class")
mean(rf.pred == train.df[, 1]) * 100 # train: 100% (check rounding)

rf.pred <- predict(rf.model, newdata=test.df, type="class")
mean(rf.pred == test.df[, 1]) * 100 # test: 91.06%


# tuning 
# best model: mtry=2
rf.tune <- train(Presence~.,
                 data=train.df,
                 method="parRF",
                 metric="ROC",
                 trControl=cv.ctrl,
                 tuneGrid=expand.grid(mtry=c(2, 3, 4, 5, 6)))

rf.tune # show tuning process
summary(rf.tune) # show results after tuning

pred <- predict(rf.tune, train.df)
confusionMatrix(pred, train.df$Presence) # train: 0.9999

pred <- predict(rf.tune, test.df)
confusionMatrix(pred, test.df$Presence) # test: 0.9079


# ROC curve
prob <- predict(rf.tune, test.df, type="prob")
ROC <- roc(response=test.df$Presence,
           predictor=prob$Yes,
           levels=levels(test.df$Presence))
plot(ROC, type="S")


# Results
# ----------------------------------------------------------------------------------------
model.list <- list(Logit=logit.tune, 
                   bayes=bayes.tune, 
                   tree=tree.tune,
                   SVM_poly=svm.poly.tune,
                   NNet=nnet.tune,
                   RF=rf.tune)

cv.values <- resamples(model.list)
dotplot(cv.values, metric="ROC")


# plot confusion matrix and ROC curve (on test set)
for (i in 1:length(model.list)) {
  # print model's name
  name <- names(model.list)[i]
  print(name) 
  
  # print confusion matrix
  pred <- predict(model.list[i], test.df)[[1]]
  mat <- caret::confusionMatrix(pred, test.df$Presence)
  print(mat$overall['Accuracy']) 
  print(mat$table)
  
  # plot ROC curve
  prob <- predict(model.list[i], test.df, type="prob")[[1]]
  ROC <- roc(response=test.df$Presence,
             predictor=prob$Yes,
             levels=levels(test.df$Presence))
  plot(ROC, type="S", add=(i!=1), col=i+1)
}

legend("bottomright", legend=names(model.list), col=2:7, lwd=2) # add legends



stopCluster(cl)
