######################################################
## DEMO FOR DETERMINING NUMBER OF CLUSTERS          ##
##  using 100 random Austin home sales in 2018-2021 ##
##  random100 (including Zone and defs).xlsx        ##
######################################################

# The following packages will be needed:
#install.packages("tidyverse")
#install.packages("cluster")
#install.packages("factoextra")
#
library(tidyverse)
library(cluster)
library(factoextra)
#
# Use Wizard to read in Excel data, or ...
library(readxl)
# Austin real estate sales: 100 randomly selected recently sold homes.
random100 <- read_excel("<YOUR FOLDER NAME HERE>\random100 (including Zone and defs).xlsx")
View(random100)
#
# We will cluster the 10 variables in columns 4-13:
rand100 <- random100[,4:13]
# Standardize each variable to mean 0, variance 1:
rand100s <- scale(rand100)
# Check appearance of standardized variables:
head(rand100s)
#
# The kmeans function does k-means CA in R.
# Determining the number of clusters is especially problematic for k-means
# since k-means lacks the natural embedding of the choice in the 
# genealogical history provided by HA clustering.
# A major reason for potential poor performance of k-means CA is a poor initial
# partition. The option nstart=25 generates 25 random initial partitions
# and kmeans reports the best of them. 

# Control the randomization by setting the seed:
set.seed(1234)
# To find 2 clusters in rand100s with 25 initial partitions:
k2 <- kmeans(rand100s, centers = 2, nstart = 25)

# The contents of the 2-cluster solution k2 may be found by clicking the 
# chevron to the left of k2 in the environment window, or by ...
structure(k2)
# Note that the output includes a calculation of the R-square of the CA = 24.0%
# Very useful parts of the CA are separately retained:
k2$cluster      # Cluster assignments of each home
k2$centers      # Centroids (means) of each cluster
k2$totss        # TSS = BSS + WSS = total variability potentially explainable by CA
k2$withinss     # WSS for each cluster
k2$tot.withinss # Sum(WSS) across all clusters (total unexplained variability)
k2$betweenss    # BSS (variability explained by the clustering)
k2$size         # Number of cases in each cluster
R2 = k2$betweenss / k2$totss
R2              # The R-square for the 2-cluster solution is 0.2403015
#
# The fviz_cluster function (in factoextra) plots the PC2 vs PC1 scores 
# of the cluster points.
p2 <- fviz_cluster(k2, geom = "point", data = rand100s) + ggtitle("k = 2")
p2 
# There is a very clear separation of the 2 clusters in terms of PC1,
# with Cl1 taking most of the homes with low PC1 scores, 
# and Cl2 taking most of the homes with high PC1 scores. 
# On the other hand, there appears to be no meaningful distinction between 
# Cl1 and Cl2 in terms of PC2.
#
# A major issue in all CA is to decide how many clusters there are.
# It is natural to try different values of k to test different solutions.
# This issue is especially acute for k-means, which lacks the genealogical 
# ties from one value of k to another that hierarchical methods have.
# Let us try k=2-7.
set.seed(1234)
k3 <- kmeans(rand100s, centers = 3, nstart = 25)
set.seed(1234)
k4 <- kmeans(rand100s, centers = 4, nstart = 25)
set.seed(1234)
k5 <- kmeans(rand100s, centers = 5, nstart = 25)
set.seed(1234)
k6 <- kmeans(rand100s, centers = 6, nstart = 25)
set.seed(1234)
k7 <- kmeans(rand100s, centers = 7, nstart = 25)

# Create corresponding plots to compare
p3 <- fviz_cluster(k3, geom = "point", data = rand100s) + ggtitle("k = 3")
p4 <- fviz_cluster(k4, geom = "point", data = rand100s) + ggtitle("k = 4")
p5 <- fviz_cluster(k5, geom = "point", data = rand100s) + ggtitle("k = 5")
p6 <- fviz_cluster(k6, geom = "point", data = rand100s) + ggtitle("k = 6")
p7 <- fviz_cluster(k7, geom = "point", data = rand100s) + ggtitle("k = 7")

# Although we can display each of p3-p7 separately, as we did p2,
# it might be informative to display all at once:
library(gridExtra)
grid.arrange(p2, p3, p4, p5, p6, p7, nrow = 2)
#
# The grid of plots shows considerable separation on the first two PCs 
# for k=2, k=3, k=4. 
# There is considerable overlap for at least 2 clusters for k>4.
# E.g., for k=7, clusters 1 and 3 largely overlap on PC1 and PC2, but the 
# other clusters are largely distinct.
# It should be remembered that separation or overlap on PCs is not the 
# same as separation or overlap on the original variables.

# Further aid can be found in the R-squares of the clusterings:
k2$betweenss / k2$totss
k3$betweenss / k3$totss
k4$betweenss / k4$totss
k5$betweenss / k5$totss
k6$betweenss / k6$totss
k7$betweenss / k7$totss
# (Note that the totss is the same value (990) for each clustering.)
# The R-squares show steady increase as k increases.
#
# Interpretation of clusters:
# The interpretation of a clustering is based upon the cluster means.
# In the plot, 4 clusters appear to be mostly separated on PC1 and PC2.
# Try interpretation of 4-cluster solution:
# Get means (centroids) of each cluster for 4-cluster solution:                   
k4$centers
# These are means of standardized vars, so hard to interpret.

# Get means of unstandardized vars:
aggregate(. ~ k4$cluster,data=rand100, FUN=mean)
# Also get cluster counts:
k4$size
#
# I will now present three methods that are commonly used to help decide 
# the number of clusters:
# Elbow method
# Average silhouette method
# Gap Statistic method
#
# Elbow method
# The "Elbow" method is similar in spirit to the scree plot of PCA. 
# The Elbow method plots a measure of clustering "(un)worthiness" against
# the number of clusters k. Commonly used is WSS. The Elbow method proposes
# as a solution for the number of clusters the value of k where the plot
# has an "elbow" - where it starts to level off - if it does.
# Control the randomization in the kmeans function by setting the seed:
set.seed(1234)
fviz_nbclust(rand100s, kmeans, method = "wss")
# WSS is a measure of error.
# It is not apparent to me that there is an "elbow" in this plot.

# Average silhouette method
# The silhouette of a case is a measure of how much more the case is like the
# other cases in its own cluster than it is like the cases in other clusters.
# The average silhouette is the mean silhouette of all cases. 
# A good clustering should have a large average silhouette.
# So plot the average silhouette against the number of clusters (or look at a table).
# The k that yields the maximum average silhouette is proposed as the 
# solution for the number of clusters.
# Control the randomization by setting the seed:
set.seed(1234)
fviz_nbclust(rand100s, kmeans, method = "silhouette")
# The silhouette method proposes k=2.
# It may be desired to interpret more clusters than 2. Most real estate
# professionals believe that more groups of homes can be distinguished. 
# The average silhouette method can justify k=4 through 7, as long as 
# meaningful interpretations can be provided for the clusters.
#
# Gap Statistic method
# A good clustering should have a small WSS.
# Compared with what? With the value you would expect by chance.
# The "Gap Statistic" method chooses k to maximize the difference ("Gap") 
# between the actual WSS and the WSS expected by chance. 
# The expected WSS is estimated by simulation.
# B is the number of simulated data sets to average to get the simulated WSS.
# K.max is the maximum number to consider for k.
# Control the randomization by setting the seed:
set.seed(1234)
gap_stat <- clusGap(rand100s, FUN = kmeans, nstart = 25,
                    K.max = 10, B = 50)
# Print the result
print(gap_stat, method = "firstmax")
fviz_gap_stat(gap_stat)
# This method points to k=1 or k=9.
