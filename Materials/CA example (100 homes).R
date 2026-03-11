## CLUSTER ANALYSIS Illustration                       ##
## Data: random100 (including Zone and defs).xlsx      ##
## Random sample of 100 Austin homes sold in 2018-2021 ##

# Install and load hierarchical agglomerative cluster analysis package
install.packages("cluster")
library(cluster)

library(readxl)
homes <- read_excel("<YOUR PATH HERE>/random100 (including Zone and defs).xlsx", 
                    sheet = "data")
# Remove four variables not used in the analysis 
homes <- homes[,5:13]

# Standardize the data
homes_scaled <- scale(homes)

# Calculate the distance matrix
dist_matrix <- dist(homes_scaled)

# Perform hierarchical clustering using Ward's original algorithm
hc <- hclust(dist_matrix, method = "ward.D2")

# Plot the dendrogram
plot(hc)

# Cut the dendrogram to get clusters
clusters <- cutree(hc, k = 6)  # Change k to the desired number of clusters

# Add cluster number to data
homes <- cbind(homes, clusters)

# Calculate the means by the cluster number

# Install and load dplyr to get means by cluster number
install.packages("dplyr")
library(dplyr)

# Calculate the means and frequencies by clustnum
means_and_counts <- summarize(
  group_by(homes, clusters),
  count = n(),
  across(where(is.numeric), mean, na.rm = TRUE)
)
# ... or (adding standard deviation)
means_and_counts <- homes %>%
  group_by(clusters) %>%
  summarize(
    count = n(), 
    across(where(is.numeric), list(mean=mean,sd=sd), na.rm = TRUE)
)

# Export results to Excel for further analysis
# Install and load rio package to get export function
install.packages("rio")
library(rio)

# Export means and counts to Excel
export(means_and_counts,"D:/means.xlsx")
