###############################################
# Hierarchical Agglomerative Cluster Analysis #
#   Stock data.xlsx                           #
###############################################
# Read in Stocks data set
# Use Data Wizard or ...
library(readxl)
stocks <- read_excel("<YOUR FOLDER LOCATION>/Stock data.xlsx")
View(stocks)

# First decide to standardize the scaling or not
# The 3 clustering vars have similar scales,
# so it does not matter much if they are standardized
# before clustering. Ordinarily, I would standardize anyway.
# But certain pedagogical points are better made with
# unstandardized data, so I will not standardize here.
# If you want to standardize, the following is an easy way:
stocksx <- as.data.frame(scale(stocks[2:4]))
# Then add the categorical vars (FIRM, Industry) back in:
stocksx <- cbind(stocksx,stocks[c(1,5)])

# Hierarchical Agglomerative (HA) clustering 
# Need matrix of pairwise distances between all cases (firms)
dist_stocks <- dist(stocks[1:3], method="euclidean")
#
# Use Ward's method of clustering
# ward.D2 is Ward's original method; ward.D is a variant.
CA.stocks <- hclust(dist_stocks,method="ward.D2")
 
# Display the dendrogram
plot(CA.stocks)
# The dendrogram displays the HA history.
# Agglomeration starts with each datapoint in its own cluster. 
# At each step, ward.D2 joins the two existing clusters that result in 
# the minimum increase in WSS (hence the maximum reduction in BSS).
# Each step corresponds to a potential cluster solution, from 1 to n.
# At the beginning there are n clusters, WSS=0, BSS=TSS, and Rsquare = 1.
# At the end there is one cluster, and WSS=TSS, BSS=0, and Rsquare = 0.
# For ward.D2, at each step WSS = 0.5*cumsum(CA.stocks$height^2).
# Therefore, from the height var, the values of WSS, BSS, and Rsquare
# can be calculated for each potential cluster solution, from 1 to n.
# This can help the user decide where to cut the tree for an HA solution.
CA.stocks$height
WSS <- 0.5*cumsum(CA.stocks$height^2)
WSS  # displays the WSS at each step in the HA
TSS <- WSS[18]
TSS
BSS <- TSS - WSS
BSS  # displays the BSS at each step
Rsq <- BSS / TSS
Rsq  # displays the R-square of the HA at each step

# Select the 3-cluster solution
cl_stocks3 = cutree(CA.stocks, k=3)
cl_stocks3  # the 3-cluster HA assignments
rect.hclust(CA.stocks,k=3)  # group the 3 clusters on the dendrogram

# Interpretation
aggregate(. ~ cl_stocks3, data=stocks[,2:4], FUN=mean)

# Add cluster number to data
stocks <- cbind(stocks, cl_stocks3)

# Plot the 3-cluster solution
plot(stocks$PROFIT, stocks$GROWTH, pch=19, 
     col=c("1"="blue","2"="red","3"="black")[cl_stocks3])

######################################################################
# Compare HA using unscaled variables with HA using scaled variables #
######################################################################
# Hierarchical Agglomerative (HA) clustering 
# Used standardized variables
# Need matrix of pairwise distances between all cases (firms)
dist_stocksx <- dist(stocksx[1:3], method="euclidean")
#
# Use Ward's method of clustering
# ward.D2 is Ward's original method; ward.D is a variant.
CA.stocksx <- hclust(dist_stocksx,method="ward.D2")

# Display the dendrogram
plot(CA.stocksx)
# The dendrogram displays the HA history.
# Agglomeration starts with each datapoint in its own cluster. 
# At each step, ward.D2 joins the two existing clusters that result in 
# the minimum increase in WSS (hence the maximum reduction in BSS).
# Each step corresponds to a potential cluster solution, from 1 to n.
# At the beginning there are n clusters, WSS=0, BSS=TSS, and Rsquare = 1.
# At the end there is one cluster, and WSS=TSS, BSS=0, and Rsquare = 0.
# For ward.D2, at each step WSS = 0.5*cumsum(CA.stocksx$height^2).
# Therefore, from the height var, the values of WSS, BSS, and Rsquare
# can be calculated for each potential cluster solution, from 1 to n.
# This can help the user decide where to cut the tree for an HA solution.
CA.stocksx$height
WSSx <- 0.5*cumsum(CA.stocksx$height^2)
WSSx  # displays the WSS at each step in the HA
TSSx <- WSSx[18]
TSSx
BSSx <- TSSx - WSSx
BSSx  # displays the BSS at each step
Rsqx <- BSSx / TSSx
Rsqx  # displays the R-square of the HA at each step

# Select the 3-cluster solution
cl_stocks3x = cutree(CA.stocksx, k=3)
cl_stocks3x  # the 3-cluster HA assignments
rect.hclust(CA.stocksx,k=3)  # group the 3 clusters on the dendrogram

# Plot the 3-cluster solution
plot(stocksx$PROFIT, stocksx$GROWTH, pch=19, 
     col=c("1"="blue","2"="red","3"="black")[cl_stocks3x])

# Interpretation in standardized scale
aggregate(. ~ cl_stocks3x, data=stocksx[,1:3], FUN=mean)

# Interpretation in original scale
aggregate(. ~ cl_stocks3x, data=stocks[,2:4], FUN=mean)

# Add cluster number to data
stocks <- cbind(stocks, cl_stocks3x)
# Which firms changed?

