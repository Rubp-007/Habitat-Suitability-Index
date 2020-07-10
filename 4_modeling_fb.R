# read in cleaned data
fb.df <- read.csv("fb_clean.csv")
dim(fb.df) # 28108 x 27
fb.df$Presence <- as.factor(fb.df$Presence)

# remove two columns
fb.df <- fb.df[, !colnames(fb.df) %in% c("BTPD_density_1km", "BTPD_density_3km")]


# separate training and test data
# using cleaned data without dimension reduction
# -------------------------------------------------
set.seed(705968)
train.id <- sample(nrow(fb.df), 16000)
train.df <- fb.df[train.id,] # 16k rows
test.df <- fb.df[-train.id,] # 12k rows


# applying models
#
# Logistic regression (not require normalization)
# -------------------------------------------------
logi.model <- glm(Presence~., data=train.df, family="binomial")
summary(logi.model)

logi.pred <- predict(logi.model, newdata=train.df, type="response")
mean(round(logi.pred) == train.df[, 1]) * 100 # train data: 78.38%

logi.pred <- predict(logi.model, newdata=test.df, type="response")
mean(round(logi.pred) == test.df[, 1]) * 100 # test data: 78.23%

# further parameter tuning: confusion matrix, ROC curve, ...


# SVM (require normalization)
# -------------------------------------------------
# install.packages("e1071")
library(e1071)
svm.model <- svm(Presence~., data=train.df, scale=TRUE, kernel="polynomial", cost=10)

svm.pred <- predict(svm.model, newdata=train.df)
mean(svm.pred == train.df[, 1]) * 100 # train: 88.78%

svm.pred <- predict(svm.model, newdata=test.df)
mean(svm.pred == test.df[, 1]) * 100 # test: 86.42%


# naive Bayesian (require normalization)
# -------------------------------------------------
bayes.model <- naiveBayes(Presence~., data=train.df)
summary(bayes.model)

bayes.pred <- predict(bayes.model, newdata=train.df)
mean(bayes.pred == train.df[, 1]) * 100 # train: 72.58%

bayes.pred <- predict(bayes.model, newdata=test.df)
mean(bayes.pred == test.df[, 1]) * 100 # test: 72.50%


# Decision Tree
# -------------------------------------------------
# install.packages("rpart")
library(rpart)
tree.model <- rpart(Presence~., data=train.df, method="class")
summary(tree.model) # can visualize the tree

tree.pred <- predict(tree.model, newdata=train.df, type="class")
mean(tree.pred == train.df[, 1]) * 100 # train: 79.24%

tree.pred <- predict(tree.model, newdata=test.df, type="class")
mean(tree.pred == test.df[, 1]) * 100 # test: 78.64%

# install.packages("rpart.plot")
library(rpart.plot)
dev.off()
rpart.plot(tree.model, type=4, faclen = 0)