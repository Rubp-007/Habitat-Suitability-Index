# read in data for location "NCCR"
nccr.df <- read.csv("nccr.csv")
View(nccr.df)
dim(nccr.df) # 55724 x 33

# columns analysis
summary(nccr.df) # mean Presence: 48.48%

#
# "Bad" columns (11):
# -------------------------------------------------
# TL2019_road_density: long small tail  ==> keep, can apply max-cut at 0.002
# nlcd16_crop_pct: long small tail      ==> 80% 0's, remove
# nlcd16_grass_pct: too flat            ==> keep
# nlcd16_forest_pct: long small tail    ==> keep, can apply max-cut
# alluvial_1km: long small tail         ==> 90% 0's, remove

# sand: peak at ~20                     ==> keep
# clay: to low at ~25                   ==> keep

# glacial_1km: too flat                 ==> always 0, remove
# colluvial_1km: long small tail        ==> almost always 0, remove

# hsp3x3: too centered                  ==> ~50% within +-20, ???
# RTNEMC: peak at ~150                  ==> ~50% of 150, ???

bad_cols <- c("TL2019_road_density", "nlcd16_crop_pct", "nlcd16_grass_pct", "nlcd16_forest_pct",
          "sand", "clay", "glacial_1km", "colluvial_1km", "alluvial_1km", "hsp3x3", "RTNEMC")


# create scatter plot to visualize correlation between bad cols and "Presence"
# -------------------------------------------------
corr.df <- nccr.df[sample(nrow(nccr.df), 500), c("Presence", bad_cols)]
View(corr.df) # 55724 x 33
summary(corr.df)
plot(corr.df)
# glacial_1km", colluvial_1km are almost always 0, so remove


# check "RTNEMC"
# -------------------------------------------------
checkStats <- function(colName) {
  vec <- unlist(nccr.df[,colName])
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
rtnemc.vec <- unlist(nccr.df[nccr.df$RTNEMC == 150, "Presence"])
print(paste0("Ratio of 150 in RTNEMC: ", length(rtnemc.vec) / nrow(nccr.df))) # 47%
print(paste0("Average Presence for 150's: ", mean(rtnemc.vec))) # 59%
# 59% > 48%, so keep it


# check "hsp3x3"
# -------------------------------------------------
checkStats("hsp3x3")
hsp.vec <- unlist(nccr.df[nccr.df$hsp3x3 > -20 & nccr.df$hsp3x3 < 20, "Presence"])
print(paste0("Ratio between -20 to 20: ", length(hsp.vec) / nrow(nccr.df))) # 51%
print(paste0("Average presence: ", mean(hsp.vec))) # 55%
# keep it or make central cut


# check "sand"
# -------------------------------------------------
checkStats("sand")
sand.vec <- unlist(nccr.df[nccr.df$sand > 15 & nccr.df$sand < 20, "Presence"])
print(paste0("Ratio between 15 to 20: ", length(sand.vec) / nrow(nccr.df))) # 40%
print(paste0("Average presence: ", mean(sand.vec))) # 57%
# should keep


# check "clay"
# -------------------------------------------------
checkStats("clay")
clay.vec <- unlist(nccr.df[nccr.df$clay < 25, "Presence"])
print(paste0("Ratio below 25: ", length(clay.vec) / nrow(nccr.df))) # 45%
print(paste0("Average presence: ", mean(clay.vec))) # 40%
# should keep


# check "TL2019_road_density"
# -------------------------------------------------
checkStats("TL2019_road_density")
road.vec <- unlist(nccr.df[nccr.df$TL2019_road_density < 0.002, "Presence"])
print(paste0("Ratio below 0.002: ", length(road.vec) / nrow(nccr.df))) # 94%
print(paste0("Average presence: ", mean(road.vec))) # 49%
# should keep this var
# apply max-cut at 0.002
road.vec <- unlist(nccr.df[nccr.df$TL2019_road_density < 0.002, "TL2019_road_density"])
dev.off()
hist(road.vec)


# check "nlcd16_grass_pct"
# -------------------------------------------------
checkStats("nlcd16_grass_pct")
sd(nccr.df$nlcd16_grass_pct) # 30
# large S.D., but okay for models


# check "nlcd16_forest_pct"
# -------------------------------------------------
checkStats("nlcd16_forest_pct")
forest.vec <- unlist(nccr.df[nccr.df$nlcd16_forest_pct == 0, "Presence"])
print(paste0("Ratio of 0's: ", length(forest.vec) / nrow(nccr.df))) # 50%
print(paste0("Average presence: ", mean(forest.vec))) # 62%
# keep it or apply max-cut


# check "alluvial_1km"
# -------------------------------------------------
checkStats("alluvial_1km")
alluvial.vec <- unlist(nccr.df[nccr.df$alluvial_1km == 0, "Presence"])
print(paste0("Ratio of 0's: ", length(alluvial.vec) / nrow(nccr.df))) # 89%
print(paste0("Average presence: ", mean(alluvial.vec))) # 50%
# too many 0's, should be removed


# removed columns
remove_cols <- c("nlcd16_crop_pct", "alluvial_1km", "glacial_1km", "colluvial_1km", "Location")
nccr_clean.df <- nccr.df[, !colnames(nccr.df) %in% remove_cols]
dim(nccr_clean.df) # 55724 x 28
View(nccr_clean.df)

# stored as cleaned data
write.csv(nccr_clean.df, "nccr_clean.csv", row.names=FALSE)






















