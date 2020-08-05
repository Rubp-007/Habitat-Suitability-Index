# read in cleaned data
nccr.df <- read.csv("nccr_clean.csv")
dim(nccr.df) # 55724 x 28

# remove two columns
nccr.df <- nccr.df[, !colnames(nccr.df) %in% c("BTPD_density_1km", "BTPD_density_3km")]
nccr.df$Presence <- as.factor(nccr.df$Presence)


# Data partitioning
# ----------------------------------------------------------------------------------------
set.seed(705968)
nccr.train.id <- sample(nrow(nccr.df), 30000)
nccr.train.df <- nccr.df[nccr.train.id,] # 30k rows
nccr.test.df <- nccr.df[-nccr.train.id,] # 25k rows


# PCA
# ----------------------------------------------------------------------------------------
# extract PCA basis from training data
nccr.pca <- prcomp(nccr.train.df[, -1], retx=TRUE, center=TRUE, scale.=TRUE)

summary(nccr.pca)
# to keep 95% variance, need to keep 16 PCs

screeplot(nccr.pca, npcs=16, type="l")
abline(h = 1, col="red")
# Since 9th PC has S.D. < 1, we need only keep 1st to 8th PCs

# save transformed training data
nccr.train.pca <- nccr.pca$x[, 1:16]

# perform same PCA transformation on test data
nccr.test.pca <- predict(nccr.pca, newdata=nccr.test.df)[, 1:16]

# visualize results of first two PCs for both training and test data
row.samp <- sample(nrow(nccr.train.pca), 1500)
plot(nccr.train.pca[row.samp, c(1,2)], 
     col=c(2,4)[nccr.train.df[row.samp, "Presence"]],
     main="NCCR first 2 PCs")

row.samp <- sample(nrow(nccr.test.pca), 1000)
points(nccr.test.pca[row.samp, c(1,2)], 
       col=c(2,4)[nccr.test.df[row.samp, "Presence"]],
       pch=16)

legend("topright", legend=levels(nccr.train.df$Presence), fill=c(2,4), border=c(2,4))

legend("bottomright", legend=c("training", "test"), col=1, pch=c(1,16))


# save PCA results for re-modeling
# ----------------------------------------------------------------------------------------
train.df <- data.frame(cbind(as.matrix(nccr.train.df$Presence), nccr.train.pca))
colnames(train.df)[1] <- "Presence"

test.df <- data.frame(cbind(as.matrix(nccr.test.df$Presence), nccr.test.pca))
colnames(test.df)[1] <- "Presence"

write.csv(train.df, "nccr_pca_train.csv", row.names=FALSE)
write.csv(test.df, "nccr_pca_test.csv", row.names=FALSE)

