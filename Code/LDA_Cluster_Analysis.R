### Add required libraries and dependencies
require(Matrix)
library(irlba)
library(topicmodels)
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

## LDA

set.seed(seed = 68)

Mlda <- LDA(M, control = list(alpha = 10, delta = .1, seed=68), k = 5, method = "Gibbs")

# User LDA scores
gamma <- Mlda@gamma

# Like LDA scores
beta <- exp(Mlda@beta)

# Log likelihood
Mlda@loglikelihood

logLik(Mlda)

# Estimate likelihood for 2-5 clusters 

lg <- list()
for (i in 2:5) {
Mlda <- LDA(M, k = i, control = list(alpha = 10, delta = .1, seed = 68), method = "Gibbs")
lg[[i]] <- logLik(Mlda) 
    }
 
plot(2:5, unlist(lg))

## LDA used for K-means Clustering

set.seed(123)
kmeans_lda <- kmeans(Mlda@gamma, 5, iter.max = 10, nstart = 3)
print(kmeans_lda)

# Plot kmeans clusters using user lda scores
fviz_cluster(kmeans_lda, data = Mlda@gamma[, -5], geom = "point", ellipse.type = "convex")

## LDA used for DBSCAN Clustering

## Used to determine optimal value of eps
kNNdistplot(Mlda@gamma, k =  5)
abline(h = 0.05, lty = 1)

dbscan_lda <- dbscan(Mlda@gamma, 0.05, 5)

# Plot dbscan clusters using user lda scores
fviz_cluster(dbscan_lda, Mlda@gamma, geom = "point")
print(dbscan_lda)

# Correlating user traits and their LDA scores (users[,-1] is used to exclude the column with IDs)
cor(gamma, users_info[,-1], use = "pairwise")

## Heat map to visualize correlations
x_heatmap <- round(cor(gamma, users_info[,-1], use="p"),2) 

y_heatmap <- melt(x_heatmap)

colnames(y_heatmap) <- c("LDA", "Trait", "r")

# Drawing the heatmap plot
qplot(x=LDA, y=Trait, data=y_heatmap, fill=r, geom="tile") +
  scale_fill_gradient2(limits=range(x_heatmap), breaks=c(min(x_heatmap), 0, max(x_heatmap)))+
  theme(axis.text=element_text(size=12), 
        axis.title=element_text(size=14,face="bold"),
        panel.background = element_rect(fill='white', colour='white'))+
  labs(x=expression('LDA'), y=NULL)

## printing 10 likes with highest LDA scores

top <- list()
for (i in 1:5) {
f <- order(beta[i,])
temp <- tail(f, n = 10) 
top[[i]]<-colnames(M)[temp]  
}

print(top)
