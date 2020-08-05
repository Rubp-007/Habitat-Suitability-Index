# read in cleaned data
fb.df <- read.csv("fb_clean.csv")
dim(fb.df) # 28108 x 27

# remove two columns
fb.df <- fb.df[, !colnames(fb.df) %in% c("BTPD_density_1km", "BTPD_density_3km")]
fb.df$Presence <- as.factor(fb.df$Presence)


# Data partitioning
# ----------------------------------------------------------------------------------------
set.seed(705968)
fb.train.id <- sample(nrow(fb.df), 15000)
fb.train.df <- fb.df[fb.train.id,] # 15k rows
fb.test.df <- fb.df[-fb.train.id,] # 13k rows


# PCA
# ----------------------------------------------------------------------------------------
# extract PCA basis from training data
fb.pca <- prcomp(fb.train.df[, -1], retx=TRUE, center=TRUE, scale.=TRUE)

summary(fb.pca)
# to keep 95% variance, need to keep 15 PCs

screeplot(fb.pca, npcs=15, type="l")
abline(h = 1, col="red")
# Since 9th PC has S.D. < 1, we need only keep 1st to 8th PCs

# save transformed training data
fb.train.pca <- fb.pca$x[, 1:15]

# perform same PCA transformation on test data
fb.test.pca <- predict(fb.pca, newdata=fb.test.df)[, 1:15]

# visualize results of first two PCs for both training and test data
row.samp <- sample(nrow(fb.train.pca), 1500)
plot(fb.train.pca[row.samp, c(1,2)], 
     col=c(2,4)[fb.train.df[row.samp, "Presence"]],
     main="fb first 2 PCs")

row.samp <- sample(nrow(fb.test.pca), 1000)
points(fb.test.pca[row.samp, c(1,2)], 
       col=c(2,4)[fb.test.df[row.samp, "Presence"]],
       pch=16)

legend("topright", legend=levels(fb.train.df$Presence), fill=c(2,4), border=c(2,4))

legend("bottomright", legend=c("training", "test"), col=1, pch=c(1,16))


# save PCA results for re-modeling
# ----------------------------------------------------------------------------------------
train.df <- data.frame(cbind(as.matrix(fb.train.df$Presence), fb.train.pca))
colnames(train.df)[1] <- "Presence"

test.df <- data.frame(cbind(as.matrix(fb.test.df$Presence), fb.test.pca))
colnames(test.df)[1] <- "Presence"

write.csv(train.df, "fb_pca_train.csv", row.names=FALSE)
write.csv(test.df, "fb_pca_test.csv", row.names=FALSE)

