# read in cleaned data
nccr.df <- read.csv("nccr_clean.csv")
View(nccr.df) # 55724 x 28
nccr.df$Presence <- as.factor(nccr.df$Presence)

# remove two columns
nccr.df <- nccr.df[, !colnames(nccr.df) %in% c("BTPD_density_1km", "BTPD_density_3km")]


# separate training and test data
# using cleaned data without dimension reduction
# -------------------------------------------------
set.seed(705968)
train.id <- sample(nrow(nccr.df), 30000)
train.df <- nccr.df[train.id,] # 30k rows
test.df <- nccr.df[-train.id,] # 25k rows


# applying models
#
# Logistic regression (not require normalization)
# -------------------------------------------------
logi.model <- glm(Presence~., data=train.df, family="binomial")
summary(logi.model)

logi.pred <- predict(logi.model, newdata=train.df, type="response")
mean(round(logi.pred) == train.df[, 1]) * 100 # train data: 81.15%

logi.pred <- predict(logi.model, newdata=test.df, type="response")
mean(round(logi.pred) == test.df[, 1]) * 100 # test data: 81.10%

# further parameter tuning: confusion matrix, ROC curve, ...


# SVM (require normalization)
# -------------------------------------------------
# install.packages("e1071")
library(e1071)
svm.model <- svm(Presence~., data=train.df, scale=TRUE, kernel="polynomial", cost=10)

svm.pred <- predict(svm.model, newdata=train.df)
mean(svm.pred == train.df[, 1]) * 100 # train: 90.87%

svm.pred <- predict(svm.model, newdata=test.df)
mean(svm.pred == test.df[, 1]) * 100 # test: 89.06%


# naive Bayesian (require normalization)
# -------------------------------------------------
bayes.model <- naiveBayes(Presence~., data=train.df)
summary(bayes.model)

bayes.pred <- predict(bayes.model, newdata=train.df)
mean(bayes.pred == train.df[, 1]) * 100 # train: 72.36%

bayes.pred <- predict(bayes.model, newdata=test.df)
mean(bayes.pred == test.df[, 1]) * 100 # test: 72.02%


# Decision Tree
# -------------------------------------------------
# install.packages("rpart")
library(rpart)
tree.model <- rpart(Presence~., data=train.df, method="class")
summary(tree.model) # can visualize the tree

tree.pred <- predict(tree.model, newdata=train.df, type="class")
mean(tree.pred == train.df[, 1]) * 100 # train: 82.08%

tree.pred <- predict(tree.model, newdata=test.df, type="class")
mean(tree.pred == test.df[, 1]) * 100 # test: 81.77%

# install.packages("rpart.plot")
library(rpart.plot)
rpart.plot(tree.model, type=4, faclen = 0)
# one branch ??

mean((nccr.df$BTPD_density_1km <= 0) != nccr.df$Presence)
