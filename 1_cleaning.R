#install.packages("readxl")
library(readxl)

# save variable dict separately
dict.df <- na.omit(read_xlsx("BTPD_data_682020.xlsx", sheet=1))
write.csv(dict.df, "dictionary.csv", row.names=FALSE)

# read in data for cleaning
# -------------------------------------------------
ori.df <- data.frame(read_xlsx("BTPD_data_682020.xlsx", sheet=2))
dim(ori.df) # 85327 x 33

# convert "Location" to factor
ori.df$Location <- as.factor(ori.df$Location)

# remove rows contains -9999
ori.df[ori.df == -9999] <- NA
ori.df <- na.omit(ori.df)
dim(ori.df) # 83832 x 33, sufficient for ML models

# split data by "Location"
nccr.df <- ori.df[ori.df$Location == "NCCR",]
fb.df <- ori.df[ori.df$Location == "FB",]


# check basic stats
# -------------------------------------------------
summary(ori.df) 
# Location: ~1/3, ~2/3, suitable for binary classification
# Presence: 50-50, suitable for binary classification

# plot histograms
plotHist <- function(df, title) {
  png(paste0("hist_", title, ".png"), width=2000, height=1600, res=150)
  par(mfcol=c(7, 5), mar=c(3,1,1,1))
  plot(df$Location, main="Location")
  for (i in 2:ncol(df)) {
    data <- df[, i]
    name <- names(df[i])
    hist(data, main=name, xlab='')
    print(paste0(name, ": ", var(data)))
  }
  dev.off()
}

plotHist(nccr.df, "nccr")
plotHist(fb.df, "fb")
plotHist(ori.df, "total")


## observation in total dataset
View(ori.df)
summary(ori.df)

# Observations
# -------------------------------------------------
# "Good" columns:
# 
# Location: 1:2, sufficient for ML models
# Presence: 50-50, suitable for binary classification
# BTPD_density_1km: many 0's
# BTPD_density_3km: many 0's
# prcp_mean9716: Gaussian
# tmax_mean9716: skewed Gaussian / Poisson
# nsrdb_dni_4km: skewed Gaussian / Poisson
# TL2019_ED_road: Poisson
# TL2019_ED_h2o: Poisson
# nlcd16_shrub_pct: not regular but okay
# silt: Gaussian
# SMAP_BTPD_grow_avg: two peaks
# SMAP_BTPD_annavg: two peaks
# rough_27x27: skewed Gaussian / Poisson
# rough_3x3: skewed Gaussian / Poisson
# hsp27x27: skewed Gaussian / Poisson
# linear_aspect: Beta
# RRTNAWS: skewed Gaussian / Poisson
# SOC_020: skewed Gaussian / Poisson
# SOC_0100: skewed Gaussian / Poisson
# SOC_030: skewed Gaussian / Poisson
# SOC_0150: skewed Gaussian / Poisson

# "Bad" columns:
#
# TL2019_road_density: long small tail
# nlcd16_crop_pct: long small tail
# nlcd16_grass_pct: too flat
# nlcd16_forest_pct: long small tail
# sand: peak at ~20 
# clay: peak at ~20
# glacial_1km: two peaks so far away
# colluvial_1km: long small tail
# alluvial_1km: long small tail
# hsp3x3: long tail
# RTNEMC: peak at ~150



# save fb.df and nccr.df as csv files for further investigation
write.csv(nccr.df, "nccr.csv", row.names=FALSE)
write.csv(fb.df, "fb.csv", row.names=FALSE)
write.csv(ori.df, "total.csv", row.names=FALSE)











  
  