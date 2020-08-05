# read in cleaned data
nccr.df <- read.csv("nccr_clean.csv")
View(nccr.df) # 55724 x 28
nccr.df$Presence <- as.factor(nccr.df$Presence)

# PCA
# -------------------------------------------------
sample.df <- nccr.df[sample(nrow(nccr.df), 1000), ]
sample.pca <- prcomp(sample.df[, -1], scale.=TRUE)
summary(sample.pca)
# ro keep 60% variance, need to keep 5 PCs
# to keep 95% variance, need to keep 17 PCs

# visualize PCs
# -------------------------------------------------
dev.off()
screeplot(sample.pca, npcs=17, type="l")
abline(h = 1, col="red")
# 9th PC has S.D. < 1, so choose 1st to 8th

# install.packages("ggfortify")
# library(ggfortify)
# autoplot(sample.pca, data=sample.df, colour="Presence", loadings=TRUE)

# install.packages("factoextra")
library(factoextra)
fviz_pca_ind(sample.pca, geom.ind="point", pointshape=21, pointsize=1, 
             col.ind=sample.df$Presence, addEllipses=TRUE, legend.title="Presence")
# first 2 PCs do not separate two classes very well...
# consider other choice: ICA, LDA1
