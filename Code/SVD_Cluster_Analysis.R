### Add required libraries and dependencies
require(Matrix)
library(irlba)
library(factoextra)
library(dbscan)
library(ggplot2)
library(reshape2)

## Load the dataset
users_info <- read.csv("../Data/users.csv")
likes_info <- read.csv("../Data/likes.csv")
ul_info <- read.csv("../Data/users-likes.csv")

# Match entries in ul_info with users_info and likes_info dictionaries
ul_info$user_row<-match(ul_info$userid,users_info$userid)
ul_info$like_row<-match(ul_info$likeid,likes_info$likeid)


head(ul_info)

## Perform Data Pre-processing

#Create a sparse User-Like Matrix M
M <- sparseMatrix(i = ul_info$user_row, j = ul_info$like_row, x = 1)

dim(M)

# Save user ids as row names
rownames(M) <- users_info$userid   

# Save like names as col_names
colnames(M) <- likes_info$name

rm(ul_info, likes_info)

# Remove rarely occuring users/Likes info - threshold - less than 50/150 times
repeat {                                       
  i <- sum(dim(M))                             
  M <- M[rowSums(M) >= 50, colSums(M) >= 150]  
  if (sum(dim(M)) == i) break                  
  }

dim(M)

# Remove the users from users_info object that were removed from M
users_info <- users_info[match(rownames(M), users_info$userid), ]

dim(users_info)

##Performing Dimensionality Reduction


## Singular Value Decomposition (SVD)

set.seed(seed = 68)

Msvd <- irlba(M, nv = 5)

# User SVD scores
u <- Msvd$u

# Like SVD scores
v <- Msvd$v

# Scree plot of SVD
plot(Msvd$d)

#  varimax-rotate the resulting SVD space:
# First obtain rotated V matrix: (unclass function has to be used to save it as an object of type matrix and not loadings)
# Perform varimax rotation:
v_rot <- unclass(varimax(Msvd$v)$loadings)
# The cross-product of M and v_rot gives u_rot:
u_rot <- as.matrix(M %*% v_rot)

## SVD used for K-means Clustering

set.seed(123)
kmeans_svd <- kmeans(Msvd$u, 5, nstart = 25)
print(kmeans_svd)

# Plot kmeans clusters using user svd scores
fviz_cluster(kmeans_svd, data = Msvd$u[, -5], geom = "point", ellipse.type = "convex")

## SVD used for DBSCAN Clustering

## Used to determine optimal value of eps

kNNdistplot(Msvd$u, k =  5)

abline(h = 0.0075, lty = 1)

dbscan_svd <- dbscan(Msvd$u, 0.0075, 5)

# Plot dbscan clusters using user svd scores
fviz_cluster(dbscan_svd, Msvd$u, geom = "point")
print(dbscan_svd)

# Correlating user traits and their SVD scores (users[,-1] is used to exclude the column with IDs)
cor(u_rot, users_info[,-1], use = "pairwise")

## Heat map to visualize correlations

x_heatmap <- round(cor(u_rot, users_info[,-1], use="p"),2)  
# Reshape it in an easy way using ggplot2
y_heatmap <- melt(x_heatmap)

colnames(y_heatmap) <- c("SVD", "Trait", "r")

# Drawing the heatmap plot
qplot(x=SVD, y=Trait, data=y_heatmap, fill=r, geom="tile") +
  scale_fill_gradient2(limits=range(x_heatmap), breaks=c(min(x_heatmap), 0, max(x_heatmap)))+
  theme(axis.text=element_text(size=12), 
        axis.title=element_text(size=14,face="bold"),
        panel.background = element_rect(fill='white', colour='white'))+
  labs(x=expression('SVD'[rot]), y=NULL)


## printing 10 likes with highest and lowest varimax-rotated SVD scores

top <- list()
bottom <-list()
for (i in 1:5) {
f <- order(v_rot[ ,i])
temp <- tail(f, n = 10)
top[[i]]<-colnames(M)[temp]  
temp <- head(f, n = 10)
bottom[[i]]<-colnames(M)[temp]  
}
print(top)
print(bottom)