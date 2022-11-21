# Read in files (provide a full path, e.g., "~/Desktop/sample_dataset/users-likes.csv")
users <- read.csv("/Users/hari/Documents/MS_coursework/Fall 2022/SWM/project/users.csv")
likes <- read.csv("/Users/hari/Documents/MS_coursework/Fall 2022/SWM/project/likes.csv")
ul <- read.csv("/Users/hari/Documents/MS_coursework/Fall 2022/SWM/project/users-likes.csv")

# Match entries in ul with users and likes dictionaries
ul$user_row<-match(ul$userid,users$userid)
ul$like_row<-match(ul$likeid,likes$likeid)

# Install Matrix library - run only once

    
# Load Matrix library
require(Matrix)
    
# Construct the sparse User-Like Matrix M
M <- sparseMatrix(i = ul$user_row, j = ul$like_row, x = 1)

# Save user IDs as row names in M
rownames(M) <- users$userid
    
# Save Like names as column names in M
colnames(M) <- likes$name
    
dim(M) #before preprocessing
# Remove ul and likes objects (they won't be needed)
rm(ul, likes)
# Remove users/Likes occurring less than 50/150 times
 
repeat {                                       
  i <- sum(dim(M))                             
  M <- M[rowSums(M) >= 50, colSums(M) >= 150]  
  if (sum(dim(M)) == i) break                  
  }

# Check the new size of M
dim(M)

# Remove the users from users object that were removed from M
users <- users[match(rownames(M), users$userid), ]

#number of users after preprocessing
dim(users)

#dimensionality reduction using svd:
# Preset the random number generator in R for the comparability of the results
set.seed(seed = 68)

# Install irlba package (run only once)


# Load irlba and extract 5 SVD dimensions
library(irlba)
Msvd <- irlba(M, nv = 5)

# User SVD scores are here:
u <- Msvd$u

# Like SVD scores are here:
v <- Msvd$v

# The scree plot of singular values:
plot(Msvd$d)

# First obtain rotated V matrix:
# (unclass function has to be used to save it as an object of type matrix and not loadings)
v_rot <- unclass(varimax(Msvd$v)$loadings)

# The cross-product of M and v_rot gives u_rot:
u_rot <- as.matrix(M %*% v_rot)

#dimensionality reduction using lda:
 #Install topicmodels package (run only once)


# Load it
library(topicmodels)

# Conduct LDA analysis, see text for details on setting
# alpha and delta parameters. 
# WARNING: this may take quite some time!

Mlda <- LDA(M, control = list(alpha = 10, delta = .1, seed=68), k = 5, method = "Gibbs")

# Extract user LDA cluster memberships
gamma <- Mlda@gamma

# Extract Like LDA clusters memberships
# betas are stored as logarithms, 
# function exp() is used to convert logs to probabilities
beta <- exp(Mlda@beta) 

# Log-likelihood of the model is stored here:
Mlda@loglikelihood

# and can be also accessed using logLik() function:
logLik(Mlda)

# Let us estimate the log-likelihood for 2,3,4, and 5 cluster solutions: 
lg <- list()
for (i in 2:5) {
Mlda <- LDA(M, k = i, control = list(alpha = 10, delta = .1, seed = 68), method = "Gibbs")
lg[[i]] <- logLik(Mlda) 
    }

plot(2:5, unlist(lg))

# Correlate user traits and their SVD scores
# users[,-1] is used to exclude the column with IDs
cor(u_rot, users[,-1], use = "pairwise")

# LDA version
cor(gamma, users[,-1], use = "pairwise")

# You need to install ggplot2 and reshape2 packages first, run only once:

    
# Load these libraries
library(ggplot2)
library(reshape2)

# Get correlations
x<-round(cor(gamma, users[,-1], use="p"),2)
   
# Reshape it in an easy way using ggplot2
y<-melt(x)
colnames(y)<-c("LDA", "Trait", "r")
    
# Produce the plot for LDA
qplot(x=LDA, y=Trait, data=y, fill=r, geom="tile") +
  scale_fill_gradient2(limits=range(x), breaks=c(min(x), 0, max(x)))+
  scale_fill_distiller(palette = "PiYG")+
  theme(axis.text=element_text(size=12), 
        axis.title=element_text(size=14,face="bold"),
        panel.background = element_rect(fill='white', colour='white'))+
  labs(x=expression('LDA'), y=NULL)
    
# Get correlations
x<-round(cor(u_rot, users[,-1], use="p"),2)
   
# Reshape it in an easy way using ggplot2
y<-melt(x)
colnames(y)<-c("SVD", "Trait", "r")
    
# Produce the plot for SVD
qplot(x=SVD, y=Trait, data=y, fill=r, geom="tile") +
  scale_fill_gradient2(limits=range(x), breaks=c(min(x), 0, max(x)))+
  scale_fill_distiller(palette = "PiYG")+
  theme(axis.text=element_text(size=12), 
        axis.title=element_text(size=14,face="bold"),
        panel.background = element_rect(fill='white', colour='white'))+
  labs(x=expression('SVD'[rot]), y=NULL)

# SVD
top_svd <- list()
bottom_svd <-list()
for (i in 1:5) {
f <- order(v_rot[ ,i])
temp <- tail(f, n = 10)
top_svd[[i]]<-colnames(M)[temp]  
temp <- head(f, n = 10)
bottom_svd[[i]]<-colnames(M)[temp]  
}

# LDA
top_lda <- list()
for (i in 1:5) {
f <- order(beta[i,])
temp <- tail(f, n = 10)
top_lda[[i]]<-colnames(M)[temp]  
}

#building prediction models:
# Split users into 10 groups
folds <- sample(1:10, size = nrow(users), replace = T)

# Take users from group 1 and assign them to the TEST subset
test <- folds == 1

# Extract SVD dimensions from the TRAINING subset
# training set can be accessed using !test
Msvd <- irlba(M[!test, ], nv = 50)

# Rotate Like SVD scores (V)
v_rot <- unclass(varimax(Msvd$v)$loadings)

# Rotate user SVD scores *for the entire sample*
u_rot <- as.data.frame(as.matrix(M %*% v_rot))

# Build linear regression model for openness
# using TRAINING subset
fit_o <- glm(users$ope~., data = u_rot, subset = !test) #glm() stands for general linear model for the data

# Inspect the regression coefficients
coef(fit_o)

# Do the same for gender
# use family = "binomial" for logistic regression model
fit_g <- glm(users$gender~.,data = u_rot, subset = !test, family = "binomial")

# Compute the predictions for the TEST subset
pred_o <- predict(fit_o, u_rot[test, ]) 
pred_g <- predict(fit_g, u_rot[test, ], type = "response")

# Correlate predicted and actual values for the TEST subset
correaltionValues <- cor(users$ope[test], pred_o)
correaltionValues

# Compute Area Under the Curve for gender
# remember to install ROCR library first
library(ROCR)
current <- prediction(pred_g, users$gender[test])
auc <- performance(current,"auc")@y.values

Msvd <- irlba(M[!test, ], nv = 150)
#examining the accuracy accross different values of k, that is for different values of dimensions:
# Choose which k are to be included in the analysis
keySet<-c(2:10,15,20,25,30,35,40,45,50,55,60)

# Preset an empty list to hold the results
resultSet <- list()

# Run the code below for each k in ks
for (k in keySet){
# Varimax rotate Like SVD dimensions 1 to k
v_rot <- unclass(varimax(Msvd$v[, 1:k])$loadings)

# This code is exactly like the one discussed earlier
u_rot <- as.data.frame(as.matrix(M %*% v_rot))
fit_o <- glm(users$ope~., data = u_rot, subset = !test)
pred_o <- predict(fit_o, u_rot[test, ])

# Save the resulting correlation coefficient as the
# element of R called k
resultSet[[as.character(k)]] <- cor(users$ope[test], pred_o)
}

# Check the results
resultSet
# Convert rs into the correct format for plotting
data<-data.frame(k=keySet, r=as.numeric(resultSet))
    
# plotfor svd vs K
ggplot(data=data, aes(x=k, y=r, group=1)) + 
    theme_light() +
    stat_smooth(colour="blue", linetype="dashed", size=1,se=F) + 
    geom_point(colour="red", size=2, shape=21, fill="white") +
    scale_y_continuous(breaks = seq(0, .5, by = 0.05))

Msvd <- irlba(M[!test, ], nv = 150)
# Start predictions
set.seed(seed=68)
n_folds<-10                # set number of folds
#kvals<- c(10,15,20,30,40,50,60,70,80,90,100,120,150)      # set k
kvals<- c(10,15,20,25,30,35,40,45,50,55,60)
vars<-colnames(users)[-1]  # choose variables to predict

folds <- sample(1:n_folds, size = nrow(users), replace = T)

resultsSVD<-list()
accuraciesSVD<-c()

resultsLDA<-list()
accuraciesLDA<-c()

for (k in kvals){
  print(k)
  for (fold in 1:n_folds){ 
    print(paste("Cross-validated predictions, fold:", fold))
    test <- folds == fold
    
    # If you want to use SVD:
    Msvd <- irlba(M[!test, ], nv = 50)
    v_rot <- unclass(varimax(Msvd$v[, 1:50])$loadings)
    predictorsSVD <- as.data.frame(as.matrix(M %*% v_rot))
    
    
    for (var in vars){
      resultsSVD[[var]]<-rep(NA, n = nrow(users))
      # check if the variable is dichotomous
      if (length(unique(na.omit(users[,var]))) ==2) {    
        fit <- glm(users[,var]~., data = predictorsSVD, subset = !test, family = "binomial")
        resultsSVD[[var]][test] <- predict(fit, predictorsSVD[test, ], type = "response")
      } else {
        fit<-glm(users[,var]~., data = predictorsSVD, subset = !test)
        resultsSVD[[var]][test] <- predict(fit, predictorsSVD[test, ])
      }
      print(paste(" Variable", var, "done."))
    }
  }



  for (fold in 1:n_folds){ 
    print(paste("Cross-validated predictions, fold:", fold))
    test <- folds == fold
    
    # If you want to use SVD:
    Msvd <- irlba(M[!test, ], nv = 50)
    v_rot <- unclass(varimax(Msvd$v[, 1:50])$loadings)
    predictorsLDA <- as.data.frame(as.matrix(M %*% v_rot))
    
    
    for (var in vars){
      resultsLDA[[var]]<-rep(NA, n = nrow(users))
      # check if the variable is dichotomous
      if (length(unique(na.omit(users[,var]))) ==2) {    
        fit <- glm(users[,var]~., data = predictorsLDA, subset = !test, family = "binomial")
        resultsLDA[[var]][test] <- predict(fit, predictorsLDA[test, ], type = "response")
      } else {
        fit<-glm(users[,var]~., data = predictorsLDA, subset = !test)
        resultsLDA[[var]][test] <- predict(fit, predictorsLDA[test, ])
      }
      print(paste(" Variable", var, "done."))
    }
  }
  
  compute_accuracy <- function(ground_truth, predicted){
    if (length(unique(na.omit(ground_truth))) ==2) {
      f<-which(!is.na(ground_truth))
      current <- prediction(predicted[f], ground_truth[f])
      return(performance(current,"auc")@y.values)
    } else {return(cor(ground_truth, predicted,use = "pairwise"))}
  }
  
  for (var in vars) 
  {
    accuraciesSVD <- c(accuraciesSVD,compute_accuracy(users[,var][test], resultsSVD[[var]][test]))
    print(var)
  }

  for (var in vars) 
  {
    accuraciesLDA <- c(accuraciesLDA,compute_accuracy(users[,var][test], resultsLDA[[var]][test]))
    print(var)
  }

  
}
print(accuraciesSVD)
print(accuraciesLDA)


traits <- c('Gender','Age','Political-Factor','Openness','Conscientiousness','Extroversion','Agreeableness','Neuroticism')
k <- c(10,10,10,10,10,10,10,10,15,15,15,15,15,15,15,15,20,20,20,20,20,20,20,20,25,25,25,25,25,25,25,25,30,30,30,30,30,30,30,30,35,35,35,35,35,35,35,35,40,40,40,40,40,40,40,40,45,45,45,45,45,45,45,45,50,50,50,50,50,50,50,50,55,55,55,55,55,55,55,55,60,60,60,60,60,60,60,60)
names <- c("SVD","SVD","SVD","SVD","SVD","SVD","SVD","SVD","LDA","LDA","LDA","LDA","LDA","LDA","LDA","LDA")

data_valsvd<-data.frame(People_Personality_Traits=traits, accuraciesSVD=as.numeric(accuraciesSVD),k=k)
# plot SVD vs K
ggplot(data_valsvd,aes(x = People_Personality_Traits, y = accuraciesSVD, group = k, color = k)) + 
  geom_line()+
  theme_light() +
  ggtitle("Predition-Accuracy with Change in Dimensions")+
  ylab(label="Reported Accuracies") + 
  xlab("People_Personality_Traits")+       
  geom_point()



data_val<-data.frame(People_Personality_Traits=traits, accuraciesLDA=as.numeric(accuraciesLDA),k=names)
# plot SVD vs LDA
library(ggplot2)
ggplot(data_val,aes(x = People_Personality_Traits, y = accuraciesLDA, group = names, color = names)) + 
  geom_line()+
  theme_light() +
  ggtitle("SVD vs LDA")+
  ylab(label="Reported Accuracies") + 
  xlab("People_Personality_Traits")+       
  geom_point()
ggsave("test1.tiff", width = 30, height = 20 , units = "cm")




