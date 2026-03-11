#############################################################
## PRINCIPAL COMPONENTS ANALYSIS                           ##
##   Rotations, orthogonal and orthonormal transformations ##
##   Stocks example                                        ##
#############################################################
# Install the necessary packages if you haven't already
install.packages("rgl")
install.packages("MASS")

# Load the packages
library(rgl)
library(MASS)

# Read in example data
library(readxl)
pca <- read_excel("<YOUR FOLDER NAME HERE>/PCA XYZ UVW ABC data and prin comps.xlsx")
View(pca)

####################
## X,Y,Z analysis ##
####################
# Create a matrix of scatterplots of X,Y,Z with solid red dots
pairs(pca[,1:3], main = "Scatterplot Matrix of X, Y, Z", pch = 19, col = "red")

# May need to force RStudio to open a 3d window for rgl:
# Open a new RGL window
open3d()

# Create a 3-dimensional scatterplot
plot3d(pca$X, pca$Y, pca$Z, col="red", size=5,  
       xlab = "X-axis", ylab = "Y-axis", zlab = "Z-axis",
       xlim = c(10,70), ylim = c(5,35), zlim = c(10,70) )
# Set the viewing angle (rotate 45 degrees around X and Y axes)
view3d(userMatrix = rotationMatrix(pi/4, 1, 0, 0) %*% rotationMatrix(pi/4, 0, 1, 0))
# Optionally, you can save the plot as an image
# rgl.snapshot("D:/3d_scatterplot.png")

####################
## U,V,W analysis ##
####################
# Create a matrix of scatterplots of U,V,W with solid red dots
pairs(pca[,4:6], main = "Scatterplot Matrix of U, V, W", pch = 19, col = "red")

# Open a new RGL window
open3d()

# Create a 3-dimensional scatterplot
plot3d(pca$U, pca$V, pca$W, col="red", size=5,  
       xlab = "U-axis", ylab = "V-axis", zlab = "W-axis")
#       xlim = c(10,70), ylim = c(5,35), zlim = c(10,70) )
# Set the viewing angle (rotate 45 degrees around U and V axes)
view3d(userMatrix = rotationMatrix(pi/4, 1, 0, 0) %*% rotationMatrix(pi/4, 0, 1, 0))

# Add an 80% confidence ellipsoid
ellipsoid <- ellipse3d(cov(pca[,4:6]), centre = colMeans(pca[,4:6]), level = 0.80)
plot3d(ellipsoid, col = "blue", alpha = 0.2, add = TRUE)

####################
## A,B,C analysis ##
####################
# Create a 3-dimensional scatterplot
open3d()
plot3d(pca$A, pca$B, pca$C, col="red", size=5,  
       xlab = "A-axis", ylab = "B-axis", zlab = "C-axis")
#       xlim = c(10,70), ylim = c(5,35), zlim = c(10,70) )
# Set the viewing angle (rotate 45 degrees around U and V axes)
view3d(userMatrix = rotationMatrix(pi/4, 1, 0, 0) %*% rotationMatrix(pi/4, 0, 1, 0))

# Add an 80% confidence ellipsoid
ellipsoid <- ellipse3d(cov(pca[,7:9]), centre = colMeans(pca[,7:9]), level = 0.80)
plot3d(ellipsoid, col = "blue", alpha = 0.2, add = TRUE)

# Calculate principal components
# The option scale.=TRUE standardizes the analysis variables
pcaXYZ <- prcomp(pca[,1:3], scale. = TRUE)
pcaUVW <- prcomp(pca[,4:6], scale. = TRUE)
pcaABC <- prcomp(pca[,7:9], scale. = TRUE)

# Compare results
pcaXYZ$x[,1] # PC1 scores
pcaUVW$x[,1] # PC1 scores
pcaABC$x[,1] # PC1 scores

# Interpretation
pcaXYZ$rotation

#########################
## Example: Stock data ##
##  "Stock data.xlsx"  ##
#########################
# read in data
library(readxl)
stocks <- read_excel("<YOUR FOLDER NAME HERE>/Stock data.xlsx")
View(stocks)

# Run PCA analysis
pcastocks <- prcomp(stocks[,2:4], scale. = TRUE)

# Interpretation
# proportions of variance explained
eigenvalues <- (pcastocks$sdev)^2
eigenvalues
eigenvalues/3
# meaning of PCs
pcastocks$rotation # eigenvectors

# Visualize the PCA dimensions
library(ggplot2)
pcastocks_pcs <- data.frame(pcastocks$x, stocks$FIRM, stocks$Industry)
ggplot(pcastocks_pcs, aes(x = PC1, y = PC2)) +
  geom_point(size = 3) +
  geom_text(aes(label = stocks.FIRM), vjust = -0.5, hjust = 0.5, size = 3, color = "black") +
  labs(title = "PCA Plot", x = "Principal Component 1", y = "Principal Component 2")

