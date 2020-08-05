# Modeling
library(caret) # train(), confusionMatrix()
library(plyr) # revalue()
library(pROC) #roc()

# 0. Data Preparation
# ----------------------------------------------------------------------------------------
nccr.df <- read.csv("nccr_clean.csv")

# remove two columns
nccr.df <- nccr.df[, !colnames(nccr.df) %in% c("BTPD_density_1km", "BTPD_density_3km")]

nccr.df$Presence <- as.factor(nccr.df$Presence)
nccr.df$Presence <- revalue(nccr.df$Presence, c("1" = "Yes", "0" = "No"))

set.seed(705968)
train.id <- sample(nrow(nccr.df), 30000)
train.df <- nccr.df[train.id,] # 30k rows
test.df <- nccr.df[-train.id,] # 25k rows

# define training parameters
cv.ctrl <- trainControl(method="cv", 
                        number=5,
                        summaryFunction=twoClassSummary,
                        classProbs=TRUE,
                        verboseIter=TRUE)


# 1. Logistic
# ----------------------------------------------------------------------------------------

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
confusionMatrix(pred, train.df$Presence) # train: 0.812 (raw) / 0.7751 (PCA_16)

pred <- predict(logit.tune, test.df)
confusionMatrix(pred, test.df$Presence) # test: 0.811 (raw) / 0.7693 (PCA_16)



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
mean(svm.pred == train.df[, 1]) * 100 # train: 90.87% (raw) / 88.62% (PCA_16)

svm.pred <- predict(svm.model, newdata=test.df)
mean(svm.pred == test.df[, 1]) * 100 # test: 89.06% (raw) / 87.37% (PCA_16)




# SVM polynomial kernel (using kernlab)
library(kernlab)
# degree = 3
# scale = 0.1
# C = 0.5

set.seed(705968)
svm.poly.tune <- train(Presence~.,
                       data=train.df,
                       method="svmPoly",
                       metric="ROC",
                       trControl=cv.ctrl)

svm.poly.tune # show tuning process
summary(svm.poly.tune) # show results after tuning

pred <- predict(svm.poly.tune, train.df)
confusionMatrix(pred, train.df$Presence) # train: 0.8942 (16)

pred <- predict(svm.poly.tune, test.df)
confusionMatrix(pred, test.df$Presence) # test: 0.8811 (16)



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
mean(bayes.pred == train.df[, 1]) * 100 # train: 75.7% (PCA_16) -> 72.36% (raw)

bayes.pred <- predict(bayes.model, newdata=test.df)
mean(bayes.pred == test.df[, 1]) * 100 # test: 75.59% (PCA_16) -> 72.02% (raw)


# tuning (parameters do not affect any results)
set.seed(705968)
bayes.tune <- train(Presence~.,
                    data=train.df,
                    method="naive_bayes",
                    metric="ROC",
                    trControl=cv.ctrl)

bayes.tune # show tuning process
summary(bayes.tune) # show results after tuning

pred <- predict(bayes.tune, train.df)
confusionMatrix(pred, train.df$Presence) # train: 0.7834 (PCA_16) -> 0.8026 (raw)

pred <- predict(bayes.tune, test.df)
confusionMatrix(pred, test.df$Presence) # test: 0.7832 (PCA_16) ->  0.80 (raw)



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

set.seed(705968)
tree.model <- rpart(Presence~., data=train.df, method="class")
summary(tree.model) # can visualize the tree

tree.pred <- predict(tree.model, newdata=train.df, type="class")
mean(tree.pred == train.df[, 1]) * 100 # train: 76.27% (PCA_16) -> 0.8208 (raw)

tree.pred <- predict(tree.model, newdata=test.df, type="class")
mean(tree.pred == test.df[, 1]) * 100 # test: 75.69% (PCA_16) -> 0.8177 (raw)


# tuning (parameters do not affect any results)
# using "rpart2" using "maxdepth" instead of "rpart" using "cp (complexity parameter)"
set.seed(705968)
tree.tune <- train(Presence~.,
                   data=train.df,
                   method="rpart2",
                   metric="ROC",
                   trControl=cv.ctrl)

tree.tune # show tuning process
summary(tree.tune) # show results after tuning

pred <- predict(tree.tune, train.df)
confusionMatrix(pred, train.df$Presence) # train: 0.747 (PCA_16) -> 0.7854 (raw)

pred <- predict(tree.tune, test.df)
confusionMatrix(pred, test.df$Presence) # test: 0.7431 (PCA_16) -> 0.7837 (raw)


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

set.seed(705968)
nnet.model <- nnet(Presence~., data=train.df, method="class", size=20)
summary(tree.model)

nnet.pred <- predict(nnet.model, newdata=train.df, type="class")
mean(nnet.pred == train.df[, 1]) * 100 # train: 84.13% (PCA_16) -> 68.73% (raw)

nnet.pred <- predict(nnet.model, newdata=test.df, type="class")
mean(nnet.pred == test.df[, 1]) * 100 # test: 83.39% (PCA_16) -> 68.39% (raw)


# tuning 
# best model: size=55, decay=1
set.seed(705968)
nnet.tune <- train(Presence~.,
                   data=train.df,
                   method="nnet",
                   metric="ROC",
                   trControl=cv.ctrl,
                   tuneGrid=expand.grid(size=c(40, 50, 55),
                                        decay=c(0, 0.5, 0.01, 0,001)))

nnet.tune # show tuning process
summary(nnet.tune) # show results after tuning

pred <- predict(nnet.tune, train.df)
confusionMatrix(pred, train.df$Presence) # train: 0.9045

pred <- predict(nnet.tune, test.df)
confusionMatrix(pred, test.df$Presence) # test: 0.8886


# ROC curve
prob <- predict(nnet.tune, test.df, type="prob")
ROC <- roc(response=test.df$Presence,
           predictor=prob$Yes,
           levels=levels(test.df$Presence))
plot(ROC, type="S")




# 6. Random Forest
# ----------------------------------------------------------------------------------------
library(randomForest)

set.seed(705968)
rf.model <- randomForest(Presence~., data=train.df)
summary(rf.model)

rf.pred <- predict(rf.model, newdata=train.df, type="class")
mean(rf.pred == train.df[, 1]) * 100 # train: 100% (check rounding)

rf.pred <- predict(rf.model, newdata=test.df, type="class")
mean(rf.pred == test.df[, 1]) * 100 # test: 91.52%


# tuning 
# best model: mtry=2
set.seed(705968)
rf.tune <- train(Presence~.,
                 data=train.df,
                 method="parRF",
                 metric="ROC",
                 trControl=cv.ctrl,
                 tuneGrid=expand.grid(mtry=c(2, 3, 4, 5, 6)))

rf.tune # show tuning process
summary(rf.tune) # show results after tuning

pred <- predict(rf.tune, train.df)
confusionMatrix(pred, train.df$Presence) # train: 0.999

pred <- predict(rf.tune, test.df)
confusionMatrix(pred, test.df$Presence) # test: 0.914


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
