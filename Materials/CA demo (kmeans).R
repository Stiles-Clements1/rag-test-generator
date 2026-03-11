#######################################################################
## CLUSTER ANALYSIS demo - kmeans                                    ##
## Data: Stock data.xlsx                                             ##
## Price/earnings ratio, profitability, growth rate for 19 companies ##
#######################################################################

# Install/load required packages/libraries
# install.packages("rgl")
library(rgl)

# Read in stocks data
library(readxl) # Or use Import Wizard
stocks <- read_excel("<YOUR PATH HERE>/Stock data.xlsx", sheet = "data")

# What clusters are there?

# An important preliminary decision is whether to cluster the variables
# in original scale or in standardized scale:
# Original: Easier to interpret
# Standardized: Prevents large-scale original variables from dominating
# Stocks vars appear roughly the same in scale, so scaling may not matter.

# Growth rate vs Profitability
plot(stocks$PROFIT, stocks$GROWTH, pch=19, col="red")

# Add clickable labels
identify(stocks$PROFIT, stocks$GROWTH, labels=stocks$FIRM)
# Label all points
text(stocks$PROFIT, stocks$GROWTH, labels=stocks$FIRM, 
     pch=19, pos=3, cex=0.7, col="blue")

# Create a matrix of scatterplots of with solid red dots
pairs(stocks[,2:4], main = "Scatterplot Matrix of Stocks Data", 
      pch = 19, col = "red")

# May need to force RStudio to open a 3d window for rgl:
# Open a new RGL window
open3d()

# Create a 3-dimensional scatterplot
plot3d(stocks$P_E, stocks$PROFIT, stocks$GROWTH, col="red", size=5,  
       xlab = "P_E", ylab = "Profit", zlab = "Growth")
#
# Try 2 clusters
k2 <- kmeans(stocks[2:4], centers = 2) 
k2$cluster               # which cluster each company is in
k2$size                  # number of companies in each cluster
plot(stocks$PROFIT, stocks$GROWTH, pch=19, 
     col=c("1"="blue","2"="red")[k2$cluster])
k2$tot.withinss          # WSS
k2$betweenss             # BSS
k2$betweenss / k2$totss  # R-square

# Try 3 clusters
k3 <- kmeans(stocks[2:4], centers = 3) 
k3$cluster
k3$size
plot(stocks$PROFIT, stocks$GROWTH, pch=19, col=c("1"="blue","2"="red","3"="black")[k3$cluster])
k3$tot.withinss
k3$betweenss
k3$betweenss / k3$totss  # R-square
#
# Try 3 clusters again. Same results?

# Kmeans algorithm:
# kmeans chooses a random subset of rows as initial "seeds".
# kmeans then assigns each row to nearest seed to get initial clustering.
# kmeans then computes centroids of initial clusters as new "seeds".
# kmeans then re-assigns rows to nearest new seed. 
# kmeans repeats above process until no rows change clusters.
#
# Consequence: Final clustering can vary since initial seeds are random.
# Deterministically setting initial seeds does not necessarily help.
# Kmeans is sensitive to choice of initial seeds.
# 
# Partial solution: Let kmeans search for the best initial seed set among
# a large group of initial seed sets: nstart option.

# Set the random number "seed" [different concept - sorry!] for reproducibility.
set.seed(1234)
k3 <- kmeans(stocks[2:4], centers = 3, nstart=50) 
k3$cluster
plot(stocks$PROFIT, stocks$GROWTH, pch=19, col=c("1"="blue","2"="red","3"="black")[k3$cluster])
k3$tot.withinss
k3$betweenss
k3$betweenss / k3$totss  # R-square

# Try 3 clusters again. Same results?

# Interpretation
pairs(stocks[,2:4], main = "Scatterplot Matrix of Stocks Data", 
      pch = 19, col=c("1"="blue","2"="red","3"="black")[k3$cluster])
k3$centers               # centroids of clusters
stocks <- cbind(k3$cluster,stocks) # which stocks are in the clusters?

# To select the number of clusters with kmeans, run kmeans with a range of 
# values for centers parm. Choose by some criterion like R-square,
# elbow, silhouette, gap (see future classes)

###########################################################
# Austin apartment rents example  #
# AustinApartmentRent.xls         #
# Need for scaling                #
###################################

# Read in apartment data
library(readxl)
apts <- read_excel("<YOUR PATH HERE>/AustinApartmentRentxls", sheet = "Sheet1")

# Compare non-scaling solution to scaled solution:
# Find 3 clusters without standardizing:
set.seed(1234)
apts3 <- kmeans(apts[2:9], centers = 3, nstart=50) 
apts3$cluster
apts3$centers

# Compare with CA on Area only:
set.seed(1234)
apts3Area <- kmeans(apts[2:2], centers = 3, nstart=50) 
apts3Area$cluster
apts3Area$centers

# The clusters are the same! (except for permuting labels 1->2, 2->3, 3->1)
# The 9-variable CA ignores all clustering vars except for Area!

# Find 3 clusters with standardizing:
aptss <- scale(apts)
set.seed(1234)
apts3s <- kmeans(aptss[,2:9], centers = 3, nstart=50) 
apts3s$cluster
apts3s$centers

# It is hard to interpret scaled clusters
# Calculate the means and frequencies of original data by cluster
aggregate(. ~ apts3s$cluster, data=apts[,2:9], FUN = mean)
apts3s$size

# Compare R-squares: unscaled vs scaled
apts3$between / apts3$totss
apts3s$between / apts3s$totss


#########################################################
# Problem: Run 6-cluster solution and interpret results
# (Following are code suggestions)
set.seed(1234)
apts6s <- kmeans(aptss[,2:9], centers = 6, nstart=50) 
apts6s$size
apts6s$centers
aggregate(. ~ apts6s$cluster, data=apts[,2:9], FUN = mean)
apts6s$between / apts6s$totss
##########################################################

