## PRICIPAL COMPONENTS ANALYSIS Illustration ##
## Data: random100(withZIP).xlsx             ##
## Random sample of 100 Austin homes sold in 2018-2021 ##

# Install and load packages for PCA
install.packages("stats")  # The 'stats' package is usually pre-installed
library(stats)
library(ggplot2) # For graphics

library(readxl)
homes <- read_excel("<YOUR PATH HERE>/random100 (including Zone and defs).xlsx", 
                    sheet = "data")
# Remove four variables not used in the analysis 
homes <- homes[,5:13]

# Standardize the data to treat all variables equally
homes_scaled <- scale(homes)

# Perform PCA
pca_result <- prcomp(homes_scaled, center = TRUE, scale. = TRUE)
    
# View the results
summary(pca_result)         # Summary of PCA results
print(pca_result$rotation)  # Principal components
print(pca_result$x)         # Transformed data (scores)

# Visualize the PCA dimensions
pca_data <- as.data.frame(pca_result$x)
ggplot(pca_data, aes(x = PC1, y = PC2)) +
  geom_point(size = 3) +
  labs(title = "PCA Plot", x = "Principal Component 1", y = "Principal Component 2")
