# read in data for location "FB"
fb.df <- read.csv("fb.csv")
dim(fb.df) # 28108 x 33
View(fb.df)

# columns analysis
summary(fb.df) # mean Presence: 54%

#
# "Bad" columns (11):
# -------------------------------------------------
# TL2019_road_density: long small tail  ==> keep, can apply max-cut at 0.002
# nlcd16_crop_pct: long small tail      ==> 76% 0's, remove
# nlcd16_grass_pct: too flat            ==> keep
# nlcd16_forest_pct: long small tail    ==> 87% 0's, remove
# alluvial_1km: long small tail         ==> 90% 0's, remove

# sand: very discrete                   ==> keep
# clay: very discrete                   ==> keep

# glacial_1km: too flat                 ==> weak correlation, remove for now
# colluvial_1km: long small tail        ==> always 0, remove

# hsp3x3: too centered                  ==> ~70% within +-20, centered, but keep
# RTNEMC: peak at ~150                  ==> ~60% of 150, but Presence ~40%, keep

bad_cols <- c("TL2019_road_density", "nlcd16_crop_pct", "nlcd16_grass_pct", "nlcd16_forest_pct",
          "sand", "clay", "glacial_1km", "colluvial_1km", "alluvial_1km", "hsp3x3", "RTNEMC")


# create scatter plot to visualize correlation between bad cols and "Presence"
# -------------------------------------------------
corr.df <- fb.df[sample(nrow(fb.df), 500), c("Presence", bad_cols)]
View(corr.df) # 500 x 12
summary(corr.df)
plot(corr.df)
# colluvial_1km always 0, so remove


# check "RTNEMC"
# -------------------------------------------------
checkStats <- function(colName) {
  vec <- unlist(fb.df[,colName])
  print(summary(vec))
  #par(mfcol=c(1,2))
  #boxplot(vec, main=colName)
  #hist(vec, main=colName, breaks=30)
  layout(mat = matrix(c(1,2),2,1, byrow=TRUE),  height = c(1,8))
  par(mar=c(0, 3.1, 1.1, 2.1))
  boxplot(vec, main=colName, horizontal=TRUE, xaxt="n", frame=F)
  par(mar=c(4, 3.1, 1.1, 2.1))
  hist(vec, xlab="", main="", breaks=40)
}
checkStats("RTNEMC")
rtnemc.vec <- unlist(fb.df[fb.df$RTNEMC == 150, "Presence"])
print(paste0("Ratio of 150 in RTNEMC: ", length(rtnemc.vec) / nrow(fb.df))) # 61%
print(paste0("Average Presence for 150's: ", mean(rtnemc.vec))) # 63%
# 63% > 54%, so keep it


# check "hsp3x3"
# -------------------------------------------------
checkStats("hsp3x3")
hsp.vec <- unlist(fb.df[fb.df$hsp3x3 > -20 & fb.df$hsp3x3 < 20, "Presence"])
print(paste0("Ratio between -20 to 20: ", length(hsp.vec) / nrow(fb.df))) # 73%
print(paste0("Average presence: ", mean(hsp.vec))) # 59%
# keep it or make central cut


# check "sand"
# -------------------------------------------------
checkStats("sand")
sand.vec <- unlist(fb.df[fb.df$sand > 40, "Presence"])
print(paste0("Ratio of data beyond 40: ", length(sand.vec) / nrow(fb.df))) # 45%
print(paste0("Average presence: ", mean(sand.vec))) # 66%
# should keep


# check "clay"
# -------------------------------------------------
checkStats("clay")
clay.vec <- unlist(fb.df[fb.df$clay > 20 & fb.df$clay < 30, "Presence"])
print(paste0("Ratio between 20 to 30: ", length(clay.vec) / nrow(fb.df))) # 56%
print(paste0("Average presence: ", mean(clay.vec))) # 61%
# should keep


# check "TL2019_road_density"
# -------------------------------------------------
checkStats("TL2019_road_density")
road.vec <- unlist(fb.df[fb.df$TL2019_road_density < 0.002, "Presence"])
print(paste0("Ratio below 0.002: ", length(road.vec) / nrow(fb.df))) # 95%
print(paste0("Average presence: ", mean(road.vec))) # 54%
# should keep this var
# can apply max-cut at 0.002
road.vec <- unlist(fb.df[fb.df$TL2019_road_density < 0.002, "TL2019_road_density"])
dev.off()
hist(road.vec)


# check "nlcd16_grass_pct"
# -------------------------------------------------
checkStats("nlcd16_grass_pct")
sd(fb.df$nlcd16_grass_pct) # 29
# large S.D., but okay for models


# check "nlcd16_forest_pct"
# -------------------------------------------------
checkStats("nlcd16_forest_pct")
forest.vec <- unlist(fb.df[fb.df$nlcd16_forest_pct == 0, "Presence"])
print(paste0("Ratio of 0's: ", length(forest.vec) / nrow(fb.df))) # 87%
print(paste0("Average presence: ", mean(forest.vec))) # 61%
# too many 0's, should remove


# check "alluvial_1km"
# -------------------------------------------------
checkStats("alluvial_1km")
alluvial.vec <- unlist(fb.df[fb.df$alluvial_1km == 0, "Presence"])
print(paste0("Ratio of 0's: ", length(alluvial.vec) / nrow(fb.df))) # 90%
print(paste0("Average presence: ", mean(hsp.vec))) # 60%
# too many 0's, should be removed


# check "glacial_1km"
# -------------------------------------------------
checkStats("glacial_1km")
glacial.vec <- unlist(fb.df[fb.df$glacial_1km == 0, "Presence"])
print(paste0("Ratio of 0's: ", length(glacial.vec) / nrow(fb.df))) # 48%
print(paste0("Average presence: ", mean(glacial.vec))) # 45%

glacial.vec <- unlist(fb.df[fb.df$glacial_1km == 1089, "Presence"])
print(paste0("Ratio of 0's: ", length(glacial.vec) / nrow(fb.df))) # 42%
print(paste0("Average presence: ", mean(glacial.vec))) # 64%
# weak correlation, either remove or cast to binomial
# remove for now


# check "nlcd16_crop_pct"
# -------------------------------------------------
checkStats("nlcd16_crop_pct")
crop.vec <- unlist(fb.df[fb.df$nlcd16_crop_pct == 0, "Presence"])
print(paste0("Ratio of 0's: ", length(crop.vec) / nrow(fb.df))) # 75%
print(paste0("Average presence: ", mean(crop.vec))) # 61%
# so many 0's, remove


# removed columns
remove_cols <- c("nlcd16_crop_pct", "nlcd16_forest_pct", "alluvial_1km", 
                 "glacial_1km", "colluvial_1km", "Location")
fb_clean.df <- fb.df[, !colnames(fb.df) %in% remove_cols]
dim(fb_clean.df) # 28108 x 27
View(fb_clean.df)

# stored as cleaned data
write.csv(fb_clean.df, "fb_clean.csv", row.names=FALSE)

