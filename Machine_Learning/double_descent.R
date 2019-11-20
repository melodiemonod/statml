
library("glmnet")
library("MASS")
library(gridExtra)
library(ggplot2)
library(doParallel)
registerDoParallel()

## Random Fourier features 



#### LOAD DATASETS ####

# Approximates Gaussian Process regression with Gaussian kernel of variance 1
# lambda: regularization parameter
# dataset: X is Nxd, y is Nx1
# D: dimensionality of random feature

# download data from http://yann.lecun.com/exdb/mnist/
download.file("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
              "train-images-idx3-ubyte.gz")
download.file("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
              "train-labels-idx1-ubyte.gz")
download.file("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
              "t10k-images-idx3-ubyte.gz")
download.file("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
              "t10k-labels-idx1-ubyte.gz")

# gunzip the files
R.utils::gunzip("train-images-idx3-ubyte.gz")
R.utils::gunzip("train-labels-idx1-ubyte.gz")
R.utils::gunzip("t10k-images-idx3-ubyte.gz")
R.utils::gunzip("t10k-labels-idx1-ubyte.gz")

# helper function for visualization
show_digit = function(arr784, col = gray(12:1 / 12), ...) {
  image(matrix(as.matrix(arr784[-785]), nrow = 28)[, 28:1], col = col, ...)
}

# load image files
load_image_file = function(filename) {
  ret = list()
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n    = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  nrow = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  ncol = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  x = readBin(f, 'integer', n = n * nrow * ncol, size = 1, signed = FALSE)
  close(f)
  data.frame(matrix(x, ncol = nrow * ncol, byrow = TRUE))
}

# load label files
load_label_file = function(filename) {
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  y = readBin(f, 'integer', n = n, size = 1, signed = FALSE)
  close(f)
  y
}

# load images
train = load_image_file("train-images-idx3-ubyte")
test  = load_image_file("t10k-images-idx3-ubyte")

# load labels
train$y = as.factor(load_label_file("train-labels-idx1-ubyte"))
test$y  = as.factor(load_label_file("t10k-labels-idx1-ubyte"))

# give matrix format
train = matrix(unlist(train), ncol = dim(train)[2], byrow = F)
test = matrix(unlist(test), ncol = dim(test)[2], byrow = F)
train = as.data.frame(train); test = as.data.frame(test)

#keep only 10,000 obs for training and 5,000 for test
set.seed(5678)
index_train = sample(1:nrow(train), 10000)
train = train[index_train,]
set.seed(89)
index_test = sample(1:nrow(test), 5000)
test = test[index_test,]

#normalize between 0 and 1
# min_features = apply(rbind(train,test), 2, min); min_f_m = matrix(nrow = nrow(train), ncol = 1,1) %*% min_features
# max_features = apply(rbind(train,test), 2, max) ; max_f_m = matrix(nrow = nrow(train), ncol = 1,1) %*% max_features
# 
# (train - matrix(nrow = nrow(train), ncol = 1,1) %*% min_features)/(matrix(nrow = nrow(train), ncol = 1,1) %*% max_features - matrix(nrow = nrow(train), ncol = 1,1) %*% min_features)
# (test - matrix(nrow = nrow(test), ncol = 1,1) %*% min_features)/(matrix(nrow = nrow(test), ncol = 1,1) %*% max_features - matrix(nrow = nrow(test), ncol = 1,1) %*% min_features)

#### DEFINE SOME VARIABLES ####

N_training = dim(train)[1]; N_test = dim(test)[1]
d = dim(train)[2] -1

#### CREATE DUMMY VARIABLES ####
create_dummy = function(data){
  data$y1 = ifelse(data[, (d+1)] == 1, 1, 0)
  data$y2 = ifelse(data[, (d+1)] == 2, 1, 0)
  data$y3 = ifelse(data[, (d+1)] == 3, 1, 0)
  data$y4 = ifelse(data[, (d+1)] == 4, 1, 0)
  data$y5 = ifelse(data[, (d+1)] == 5, 1, 0)
  data$y6 = ifelse(data[, (d+1)] == 6, 1, 0)
  data$y7 = ifelse(data[, (d+1)] == 7, 1, 0)
  data$y8 = ifelse(data[, (d+1)] == 8, 1, 0)
  data$y9 = ifelse(data[, (d+1)] == 9, 1, 0)
  data$y10 = ifelse(data[, (d+1)] == 10, 1, 0)
  return(data)
  }

train = create_dummy(train)
test = create_dummy(test)

#### FIND OPTIMUM LAMBDA FOR LEAST SQUARE SUBJECT TO MIN L2 NORM ####

# intermediary features
train = as.matrix(train); test = as.matrix(test)
D = 1000 # number of random fourrier features
set.seed(987)
v = matrix(nrow = d, ncol = D, rnorm(D*d))
b = 2 * pi * runif(D, 0, 1);
Z = sqrt(2/D)*cos(train[, 1:d] %*% v + matrix(nrow = N_training, ncol = 1,1) %*% b);

lambda_grid = 10^seq(-20, -6, 1)
# true parameters 
diff = foreach(i = 1:10) %dopar% {
  apply(matrix(ncol = length(lambda_grid), byrow = F, rep(ginv(Z) %*% train[, d+1+i], length(lambda_grid))) - coef(cv.glmnet(x = Z, y = train[,d+1+i], alpha = 0, lambda = lambda_grid, intercept = F), s = lambda_grid)[-1,], 2, function(x) sum(x)^2)  
}

lambda_ls = numeric(10)
for(i in 1:10){
  lambda_ls[i] = lambda_grid[which.min(diff[[i]])]
}

#[1] [1] 1e-20 1e-16 1e-16 1e-18 1e-16 1e-16 1e-17 1e-15 1e-20 1e-16
#lambda_ls = c(1e-20, 1e-16, 1e-16, 1e-18, 1e-16, 1e-16, 1e-17 ,1e-15, 1e-20, 1e-16)

#logit regression
W = prop.table(table(train[, d+1]))*(1 - prop.table(table(train[, d+1])))
alpha.ls = foreach(i = 1:10) %dopar%{
  ginv(diag(rep(sqrt(W[i]), N_training)) %*% Z) %*% diag(rep(sqrt(W[i]), N_training)) %*% train[, d+1+i]
}

alpha.ls[[6]] =   ginv(diag(rep(sqrt(W[6]), N_training)) %*% Z) %*% diag(rep(sqrt(W[6]), N_training)) %*% train[, d+1+6]

lambda_grid = 10^seq(-50,-20, 1)
diff2 = matrix(ncol = length(lambda_grid), nrow = 10, numeric())
for(i in 1:10){ 
  diff2[i,] = apply(matrix(ncol = length(lambda_grid), byrow = F, rep(alpha.ls[[i]], length(lambda_grid))) - coef(cv.glmnet(x = Z, y = train[,d+1+i], alpha = 0, lambda = lambda_grid, intercept = F, family = "binomial"), s = lambda_grid)[-1,], 2, function(x) sum(x)^2)  
}
lambda_ls2 = apply(diff2, 1, function(x) lambda_grid[which.min(x)]) 
#  [1] 1e-20 1e-16 1e-15 1e-17 1e-16 1e-20 1e-20 1e-15 1e-18 1e-16
# lambda_ls2 =  c(1e-20, 1e-16, 1e-15, 1e-17, 1e-16, 1e-20, 1e-20, 1e-15, 1e-18, 1e-16)

#### Prepare intermediary features ####
## RUN ON CLUSTER
# D_max = 20*10^3
# set.seed(987)
# v = matrix(nrow = d, ncol = D_max, rnorm(D_max*d))
# b = 2 * pi * runif(D_max, 0, 1);
# Z_tot = cos(train[, -(d+1)] %*% v + matrix(nrow = N_training, ncol = 1,1) %*% b);
# Z_test_tot = cos(test[, -(d+1)] %*% v + matrix(nrow = N_test, ncol = 1,1) %*% b);
# 
# save(Z_tot, file = file.path("/rds/general/user/mm3218/home/res/Z_tot"))
# save(Z_test_tot, file = file.path("/rds/general/user/mm3218/home/res/Z_test_tot"))

# load(file = file.path("/rds/general/user/mm3218/home/res/Z_tot"))
# load(file = file.path("/rds/general/user/mm3218/home/res/Z_test_tot"))

load(file = file.path("~/Documents/PhD/Modules/Machine_Learning/code/Z_tot"))
load(file = file.path("~/Documents/PhD/Modules/Machine_Learning/code/Z_test_tot"))



#### REGRESSION - CODE ####

reg = function(D, train, test){
  
  # intermediary features (RFF)
  Z = sqrt(2/D)*Z_tot[, 1:D]
  Z_test = sqrt(2/D)*Z_test_tot[, 1:D]
  
  # Square loss function subject to minimum L2 norm (using small L2 penalty)
  glmnet.ls = function(i){
    glmnet(x = Z, y = train[,d+1+i], alpha = 1, lambda = lambdas, intercept = F, family = "binomial")
  }
  cv_fit.ls = mclapply(1:10, function(x) glmnet.ls(x), mc.cores = 1)
  alpha_ls = cbind(coef(cv_fit.ls[[1]]), coef(cv_fit.ls[[2]]), coef(cv_fit.ls[[3]]), coef(cv_fit.ls[[4]]), coef(cv_fit.ls[[5]]), coef(cv_fit.ls[[6]]), coef(cv_fit.ls[[7]]), coef(cv_fit.ls[[8]]), coef(cv_fit.ls[[9]]), coef(cv_fit.ls[[10]]))
  L2_norm.ls = sum(apply(alpha_ls, 1, mean)^2)
  prob_training.ls = cbind(predict(cv_fit.ls[[1]], newx = Z), predict(cv_fit.ls[[2]], newx = Z), predict(cv_fit.ls[[3]], newx = Z), predict(cv_fit.ls[[4]], newx = Z), predict(cv_fit.ls[[5]], newx = Z), predict(cv_fit.ls[[6]], newx = Z), predict(cv_fit.ls[[7]], newx = Z),
                        predict(cv_fit.ls[[8]], newx = Z), predict(cv_fit.ls[[9]], newx = Z), predict(cv_fit.ls[[10]], newx = Z))
  prob_test.ls = cbind(predict(cv_fit.ls[[1]], newx = Z_test), predict(cv_fit.ls[[2]], newx = Z_test), predict(cv_fit.ls[[3]], newx = Z_test), predict(cv_fit.ls[[4]], newx = Z_test), predict(cv_fit.ls[[5]], newx = Z_test), predict(cv_fit.ls[[6]], newx = Z_test), 
                       predict(cv_fit.ls[[7]], newx = Z_test), predict(cv_fit.ls[[8]], newx = Z_test), predict(cv_fit.ls[[9]], newx = Z_test), predict(cv_fit.ls[[10]], newx = Z_test))
  MSE_train.ls = mean(as.matrix(train[, (d+2):(d+11)] - prob_training.ls)^2)
  MSE_test.ls = mean(as.matrix(train[, (d+2):(d+11)] - prob_test.ls)^2)
  SE_train.ls = sum((train[, (d+2):(d+11)] - prob_training.ls)^2)
  SE_test.ls = sum((train[, (d+2):(d+11)] - prob_test.ls)^2)
  ZOL_train.ls = mean(as.integer(train[,d+1]) != apply(prob_training.ls, 1, which.max))
  ZOL_test.ls = mean(as.integer(test[,d+1]) != apply(prob_test.ls, 1, which.max))
  
  # Ridge
  lambdas = c(1000, 100,   30,   10,   3,    1,    0.3,    0.1,    0.03,    0.01, 0.001, 0.0001)
  cv.glmnet.ridge = function(i){
    cv.glmnet(x = Z, y = train[,d+1+i], alpha = 0, lambda = lambdas, intercept = F, family = "binomial")
  }
  cv_fit.ridge = mclapply(1:10, function(x) cv.glmnet.ridge(x), mc.cores = 1)
  lambda_min.ridge <- c(cv_fit.ridge[[1]]$lambda.min, cv_fit.ridge[[2]]$lambda.min, cv_fit.ridge[[3]]$lambda.min, cv_fit.ridge[[4]]$lambda.min, cv_fit.ridge[[5]]$lambda.min, cv_fit.ridge[[6]]$lambda.min, cv_fit.ridge[[7]]$lambda.min, 
                        cv_fit.ridge[[8]]$lambda.min, cv_fit.ridge[[9]]$lambda.min, cv_fit.ridge[[10]]$lambda.min)
  L2_norm.ridge = sapply(lambdas, function(x) sum(apply(cbind(coef(cv_fit.ridge[[1]], s =x), coef(cv_fit.ridge[[2]], s =x), coef(cv_fit.ridge[[3]], s =x), coef(cv_fit.ridge[[4]], s =x), coef(cv_fit.ridge[[5]], s =x), coef(cv_fit.ridge[[6]], s =x), coef(cv_fit.ridge[[7]], s =x), 
                                                              coef(cv_fit.ridge[[8]], s =x), coef(cv_fit.ridge[[9]], s =x), coef(cv_fit.ridge[[10]], s =x)), 1, mean)^2))
  prob_training.ridge = function(y) cbind(predict(cv_fit.ridge[[1]], newx = Z, s = y), predict(cv_fit.ridge[[2]], newx = Z, s = y), predict(cv_fit.ridge[[3]], newx = Z, s = y), predict(cv_fit.ridge[[4]], newx = Z, s = y), predict(cv_fit.ridge[[5]], newx = Z, s = y), 
                                       predict(cv_fit.ridge[[6]], newx = Z, s = y), predict(cv_fit.ridge[[7]], newx = Z, s = y), predict(cv_fit.ridge[[8]], newx = Z, s = y), predict(cv_fit.ridge[[9]], newx = Z, s = y), predict(cv_fit.ridge[[10]], newx = Z, s = y))
  prob_test.ridge = function(y) cbind(predict(cv_fit.ridge[[1]], newx = Z_test, s = y), predict(cv_fit.ridge[[2]], newx = Z_test, s = y), predict(cv_fit.ridge[[3]], newx = Z_test, s = y), predict(cv_fit.ridge[[4]], newx = Z_test, s = y), predict(cv_fit.ridge[[5]], newx = Z_test, s = y), 
                       predict(cv_fit.ridge[[6]], newx = Z_test), predict(cv_fit.ridge[[7]], newx = Z_test, s = y), predict(cv_fit.ridge[[8]], newx = Z_test, s = y), predict(cv_fit.ridge[[9]], newx = Z_test, s = y), predict(cv_fit.ridge[[10]], newx = Z_test, s = y))
  ZOL_train.ridge = sapply(lambdas, function(x) mean(as.integer(train[,d+1]) != apply(prob_training.ridge(x), 1, which.max)))
  ZOL_test.ridge = sapply(lambdas, function(x) mean(as.integer(test[,d+1]) != apply(prob_test.ridge(x), 1, which.max)))
  MSE_train.ridge = sapply(lambdas, function(x) mean(as.matrix(train[, (d+2):(d+11)] - prob_training.ridge(x))^2))
  MSE_test.ridge = sapply(lambdas, function(x) mean(as.matrix(test[, (d+2):(d+11)] - prob_test.ridge(x))^2))
  SE_train.ridge = sapply(lambdas, function(x) sum((train[, (d+2):(d+11)] - prob_training.ridge(x))^2))
  SE_test.ridge = sapply(lambdas, function(x) sum((test[, (d+2):(d+11)] - prob_test.ridge(x))^2))
  
  # Lasso
  cv.glmnet.lasso = function(i){
    cv.glmnet(x = Z, y = train[,d+1+i], alpha = 1, lambda = lambdas, intercept = F, family = "binomial")
  }
  cv_fit.lasso = mclapply(1:10, function(x) cv.glmnet.lasso(x), mc.cores = 1)
  lambda_min.lasso <- c(cv_fit.lasso[[1]]$lambda.min, cv_fit.lasso[[2]]$lambda.min, cv_fit.lasso[[3]]$lambda.min, cv_fit.lasso[[4]]$lambda.min, cv_fit.lasso[[5]]$lambda.min, cv_fit.lasso[[6]]$lambda.min, cv_fit.lasso[[7]]$lambda.min, 
                        cv_fit.lasso[[8]]$lambda.min, cv_fit.lasso[[9]]$lambda.min, cv_fit.lasso[[10]]$lambda.min)
  L2_norm.lasso = sapply(lambdas, function(x) sum(apply(cbind(coef(cv_fit.lasso[[1]], s =x), coef(cv_fit.lasso[[2]], s =x), coef(cv_fit.lasso[[3]], s =x), coef(cv_fit.lasso[[4]], s =x), coef(cv_fit.lasso[[5]], s =x), coef(cv_fit.lasso[[6]], s =x), coef(cv_fit.lasso[[7]], s =x), 
                                                              coef(cv_fit.lasso[[8]], s =x), coef(cv_fit.lasso[[9]], s =x), coef(cv_fit.lasso[[10]], s =x)), 1, mean)^2))
  prob_training.lasso = function(y) cbind(predict(cv_fit.lasso[[1]], newx = Z, s = y), predict(cv_fit.lasso[[2]], newx = Z, s = y), predict(cv_fit.lasso[[3]], newx = Z, s = y), predict(cv_fit.lasso[[4]], newx = Z, s = y), predict(cv_fit.lasso[[5]], newx = Z, s = y), 
                                       predict(cv_fit.lasso[[6]], newx = Z, s = y), predict(cv_fit.lasso[[7]], newx = Z, s = y), predict(cv_fit.lasso[[8]], newx = Z, s = y), predict(cv_fit.lasso[[9]], newx = Z, s = y), predict(cv_fit.lasso[[10]], newx = Z, s = y))
  prob_test.lasso = function(y) cbind(predict(cv_fit.lasso[[1]], newx = Z_test, s = y), predict(cv_fit.lasso[[2]], newx = Z_test, s = y), predict(cv_fit.lasso[[3]], newx = Z_test, s = y), predict(cv_fit.lasso[[4]], newx = Z_test, s = y), predict(cv_fit.lasso[[5]], newx = Z_test, s = y), 
                                   predict(cv_fit.lasso[[6]], newx = Z_test), predict(cv_fit.lasso[[7]], newx = Z_test, s = y), predict(cv_fit.lasso[[8]], newx = Z_test, s = y), predict(cv_fit.lasso[[9]], newx = Z_test, s = y), predict(cv_fit.lasso[[10]], newx = Z_test, s = y))
  ZOL_train.lasso = sapply(lambdas, function(x) mean(as.integer(train[,d+1]) != apply(prob_training.lasso(x), 1, which.max)))
  ZOL_test.lasso = sapply(lambdas, function(x) mean(as.integer(test[,d+1]) != apply(prob_test.lasso(x), 1, which.max)))
  MSE_train.lasso = sapply(lambdas, function(x) mean(as.matrix(train[, (d+2):(d+11)] - prob_training.lasso(x))^2))
  MSE_test.lasso = sapply(lambdas, function(x) mean(as.matrix(test[, (d+2):(d+11)] - prob_test.lasso(x))^2))
  SE_train.lasso = sapply(lambdas, function(x) sum((train[, (d+2):(d+11)] - prob_training.lasso(x))^2))
  SE_test.lasso = sapply(lambdas, function(x) sum((test[, (d+2):(d+11)] - prob_test.lasso(x))^2))
  
  return(list(ls = list(L2_norm = L2_norm.ls,
                        ZOL_train = ZOL_train.ls, ZOL_test = ZOL_test.ls, MSE_train = MSE_train.ls, MSE_test = MSE_test.ls, SE_train = SE_train.ls, SE_test = SE_test.ls),
              ridge = list(lambda_min = lambda_min.ridge,
                           L2_norm = L2_norm.ridge,
                           ZOL_train = ZOL_train.ridge, ZOL_test = ZOL_test.ridge, MSE_train = MSE_train.ridge, MSE_test = MSE_test.ridge, SE_train = SE_train.ridge, SE_test = SE_test.ridge),
              lasso = list(lambda_min = lambda_min.lasso,
                           L2_norm = L2_norm.lasso,
                           ZOL_train = ZOL_train.lasso, ZOL_test = ZOL_test.lasso, MSE_train = MSE_train.lasso, MSE_test = MSE_test.lasso, SE_train = SE_train.lasso, SE_test = SE_test.lasso)))
}

reg2 = function(D, train, test){
  
  # intermediary features
  Z = sqrt(2/D)*Z_tot[, 1:D]
  Z_test = sqrt(2/D)*Z_test_tot[, 1:D]
  
  # Square loss function subject to minimum L2 norm (using Moore–Penrose inverse)
  
  alpha_lsf = function(i){
    ginv(Z) %*% train[, d+1+i]
  }
  alpha_ls = mclapply(1:10, function(x) alpha_lsf(x), mc.cores = 1)
  
  prob_training.lsf = function(i){
    Z%*%alpha_ls[[i]]
  }
  prob_training.ls = mclapply(1:10, function(x) prob_training.lsf(x), mc.cores = 1)
  
  prob_test.lsf = function(i){
    Z_test%*%alpha_ls[[i]]
  }
  prob_test.ls = mclapply(1:10, function(x) prob_test.lsf(x), mc.cores = 1)

  L2_norm.ls = sum(apply(matrix(unlist(alpha_ls), nrow = D, byrow = F), 1, mean)^2)
  MSE_train.ls = mean(as.matrix(train[, (d+2):(d+11)] - matrix(unlist(prob_training.ls), nrow = N_training, byrow = F))^2)
  MSE_test.ls = mean(as.matrix(test[, (d+2):(d+11)] -  matrix(unlist(prob_test.ls), nrow = N_test, byrow = F))^2)
  SE_train.ls = sum((train[, (d+2):(d+11)] - matrix(unlist(prob_training.ls), nrow = N_training, byrow = F))^2)
  SE_test.ls = sum((test[, (d+2):(d+11)] -  matrix(unlist(prob_test.ls), nrow = N_test, byrow = F))^2)
  ZOL_train.ls = mean(as.integer(train[,d+1]) != apply(matrix(unlist(prob_training.ls), nrow = N_training, byrow = F), 1, which.max))
  ZOL_test.ls = mean(as.integer(test[,d+1]) != apply(matrix(unlist(prob_test.ls), nrow = N_test, byrow = F), 1, which.max))
  
  return(list(ls = list(L2_norm = L2_norm.ls, ZOL_train = ZOL_train.ls, ZOL_test = ZOL_test.ls, MSE_train = MSE_train.ls, MSE_test = MSE_test.ls, SE_train = SE_train.ls, SE_test = SE_test.ls)))
}

reg3 = function(D, train, test){
  
  # intermediary features (RFF)
  Z = sqrt(2/D)*Z_tot[, 1:D]
  Z_test = sqrt(2/D)*Z_test_tot[, 1:D]
  
  # Square loss function subject to minimum L2 norm (using small L2 penalty)
  glmnet.ls = function(i){
    glmnet(x = Z, y = train[,d+1+i], alpha = 1, lambda = lambdas, intercept = F, family = "binomial")
  }
  cv_fit.ls = mclapply(1:10, function(x) glmnet.ls(x), mc.cores = 1)
  alpha_ls = cbind(coef(cv_fit.ls[[1]]), coef(cv_fit.ls[[2]]), coef(cv_fit.ls[[3]]), coef(cv_fit.ls[[4]]), coef(cv_fit.ls[[5]]), coef(cv_fit.ls[[6]]), coef(cv_fit.ls[[7]]), coef(cv_fit.ls[[8]]), coef(cv_fit.ls[[9]]), coef(cv_fit.ls[[10]]))
  L2_norm.ls = sum(apply(alpha_ls, 1, mean)^2)
  prob_training.ls = cbind(predict(cv_fit.ls[[1]], newx = Z, type="response"), predict(cv_fit.ls[[2]], newx = Z, type="response"), predict(cv_fit.ls[[3]], newx = Z, type="response"), predict(cv_fit.ls[[4]], newx = Z, type="response"), 
                           predict(cv_fit.ls[[5]], newx = Z, type="response"), predict(cv_fit.ls[[6]], newx = Z, type="response"), predict(cv_fit.ls[[7]], newx = Z, type="response"),
                           predict(cv_fit.ls[[8]], newx = Z, type="response"), predict(cv_fit.ls[[9]], newx = Z, type="response"), predict(cv_fit.ls[[10]], newx = Z, type="response"))
  prob_test.ls = cbind(predict(cv_fit.ls[[1]], newx = Z_test, type="response"), predict(cv_fit.ls[[2]], newx = Z_test, type="response"), predict(cv_fit.ls[[3]], newx = Z_test, type="response"), predict(cv_fit.ls[[4]], newx = Z_test, type="response"),
                       predict(cv_fit.ls[[5]], newx = Z_test, type="response"), predict(cv_fit.ls[[6]], newx = Z_test, type="response"), 
                       predict(cv_fit.ls[[7]], newx = Z_test, type="response"), predict(cv_fit.ls[[8]], newx = Z_test, type="response"), predict(cv_fit.ls[[9]], newx = Z_test, type="response"), predict(cv_fit.ls[[10]], newx = Z_test, type="response"))
  MSE_train.ls = mean(as.matrix(train[, (d+2):(d+11)] - prob_training.ls)^2)
  MSE_test.ls = mean(as.matrix(train[, (d+2):(d+11)] - prob_test.ls)^2)
  SE_train.ls = sum((train[, (d+2):(d+11)] - prob_training.ls)^2)
  SE_test.ls = sum((train[, (d+2):(d+11)] - prob_test.ls)^2)
  ZOL_train.ls = mean(as.integer(train[,d+1]) != apply(prob_training.ls, 1, which.max))
  ZOL_test.ls = mean(as.integer(test[,d+1]) != apply(prob_test.ls, 1, which.max))
  
  # Ridge
  lambdas = c(1000, 100,  30, 10, 3, 1, 0.3, 0.1, 0.03, 0.01, 0.001, 0.0001)
  cv.glmnet.ridge = function(i){
    cv.glmnet(x = Z, y = train[,d+1+i], alpha = 0, lambda = lambdas, intercept = F, family = "binomial")
  }
  cv_fit.ridge = mclapply(1:10, function(x) cv.glmnet.ridge(x), mc.cores = 1)
  
  lambda_min.ridge <- c(cv_fit.ridge[[1]]$lambda.min, cv_fit.ridge[[2]]$lambda.min, cv_fit.ridge[[3]]$lambda.min, cv_fit.ridge[[4]]$lambda.min, cv_fit.ridge[[5]]$lambda.min, cv_fit.ridge[[6]]$lambda.min, cv_fit.ridge[[7]]$lambda.min, 
                        cv_fit.ridge[[8]]$lambda.min, cv_fit.ridge[[9]]$lambda.min, cv_fit.ridge[[10]]$lambda.min)
  L2_norm.ridge = sapply(lambdas, function(x) sum(apply(cbind(coef(cv_fit.ridge[[1]], s =x), coef(cv_fit.ridge[[2]], s =x), coef(cv_fit.ridge[[3]], s =x), coef(cv_fit.ridge[[4]], s =x), coef(cv_fit.ridge[[5]], s =x), coef(cv_fit.ridge[[6]], s =x), coef(cv_fit.ridge[[7]], s =x), 
                                                              coef(cv_fit.ridge[[8]], s =x), coef(cv_fit.ridge[[9]], s =x), coef(cv_fit.ridge[[10]], s =x)), 1, mean)^2))
  prob_training.ridge = function(y) cbind(predict(cv_fit.ridge[[1]], newx = Z, s = y, type="response"), predict(cv_fit.ridge[[2]], newx = Z, s = y, type="response"), predict(cv_fit.ridge[[3]], newx = Z, s = y, type="response"),
                                          predict(cv_fit.ridge[[4]], newx = Z, s = y, type="response"), predict(cv_fit.ridge[[5]], newx = Z, s = y, type="response"), 
                                          predict(cv_fit.ridge[[6]], newx = Z, s = y, type="response"), predict(cv_fit.ridge[[7]], newx = Z, s = y, type="response"), predict(cv_fit.ridge[[8]], newx = Z, s = y, type="response"),
                                          predict(cv_fit.ridge[[9]], newx = Z, s = y, type="response"), predict(cv_fit.ridge[[10]], newx = Z, s = y, type="response"))
  prob_test.ridge = function(y) cbind(predict(cv_fit.ridge[[1]], newx = Z_test, s = y, type="response"), predict(cv_fit.ridge[[2]], newx = Z_test, s = y, type="response"), predict(cv_fit.ridge[[3]], newx = Z_test, s = y, type="response"), 
                                      predict(cv_fit.ridge[[4]], newx = Z_test, s = y, type="response"), predict(cv_fit.ridge[[5]], newx = Z_test, s = y, type="response"), 
                                      predict(cv_fit.ridge[[6]], newx = Z_test, s = y, type="response"), predict(cv_fit.ridge[[7]], newx = Z_test, s = y, type="response"), predict(cv_fit.ridge[[8]], newx = Z_test, s = y, type="response"), 
                                      predict(cv_fit.ridge[[9]], newx = Z_test, s = y, type="response"), predict(cv_fit.ridge[[10]], newx = Z_test, s = y, type="response"))
  ZOL_train.ridge = sapply(lambdas, function(x) mean(as.integer(train[,d+1]) != apply(prob_training.ridge(x), 1, which.max)))
  ZOL_test.ridge = sapply(lambdas, function(x) mean(as.integer(test[,d+1]) != apply(prob_test.ridge(x), 1, which.max)))
  MSE_train.ridge = sapply(lambdas, function(x) mean(as.matrix(train[, (d+2):(d+11)] - prob_training.ridge(x))^2))
  MSE_test.ridge = sapply(lambdas, function(x) mean(as.matrix(test[, (d+2):(d+11)] - prob_test.ridge(x))^2))
  SE_train.ridge = sapply(lambdas, function(x) sum((train[, (d+2):(d+11)] - prob_training.ridge(x))^2))
  SE_test.ridge = sapply(lambdas, function(x) sum((test[, (d+2):(d+11)] - prob_test.ridge(x))^2))
  
  # Lasso
  cv.glmnet.lasso = function(i){
    cv.glmnet(x = Z, y = train[,d+1+i], alpha = 1, lambda = lambdas, intercept = F, family = "binomial")
  }
  cv_fit.lasso = mclapply(1:10, function(x) cv.glmnet.lasso(x), mc.cores = 1)
  lambda_min.lasso <- c(cv_fit.lasso[[1]]$lambda.min, cv_fit.lasso[[2]]$lambda.min, cv_fit.lasso[[3]]$lambda.min, cv_fit.lasso[[4]]$lambda.min, cv_fit.lasso[[5]]$lambda.min, cv_fit.lasso[[6]]$lambda.min, cv_fit.lasso[[7]]$lambda.min, 
                        cv_fit.lasso[[8]]$lambda.min, cv_fit.lasso[[9]]$lambda.min, cv_fit.lasso[[10]]$lambda.min)
  L2_norm.lasso = sapply(lambdas, function(x) sum(apply(cbind(coef(cv_fit.lasso[[1]], s =x), coef(cv_fit.lasso[[2]], s =x), coef(cv_fit.lasso[[3]], s =x), coef(cv_fit.lasso[[4]], s =x), coef(cv_fit.lasso[[5]], s =x), coef(cv_fit.lasso[[6]], s =x), coef(cv_fit.lasso[[7]], s =x), 
                                                              coef(cv_fit.lasso[[8]], s =x), coef(cv_fit.lasso[[9]], s =x), coef(cv_fit.lasso[[10]], s =x)), 1, mean)^2))
  prob_training.lasso = function(y) cbind(predict(cv_fit.lasso[[1]], newx = Z, s = y, type="response"), predict(cv_fit.lasso[[2]], newx = Z, s = y, type="response"), predict(cv_fit.lasso[[3]], newx = Z, s = y, type="response"), 
                                          predict(cv_fit.lasso[[4]], newx = Z, s = y, type="response"), predict(cv_fit.lasso[[5]], newx = Z, s = y, type="response"), 
                                          predict(cv_fit.lasso[[6]], newx = Z, s = y, type="response"), predict(cv_fit.lasso[[7]], newx = Z, s = y, type="response"), predict(cv_fit.lasso[[8]], newx = Z, s = y, type="response"), 
                                          predict(cv_fit.lasso[[9]], newx = Z, s = y, type="response"), predict(cv_fit.lasso[[10]], newx = Z, s = y, type="response"))
  prob_test.lasso = function(y) cbind(predict(cv_fit.lasso[[1]], newx = Z_test, s = y, type="response"), predict(cv_fit.lasso[[2]], newx = Z_test, s = y, type="response"), predict(cv_fit.lasso[[3]], newx = Z_test, s = y, type="response"),
                                      predict(cv_fit.lasso[[4]], newx = Z_test, s = y, type="response"), predict(cv_fit.lasso[[5]], newx = Z_test, s = y, type="response"), 
                                      predict(cv_fit.lasso[[6]], newx = Z_test, s=y,  type="response"), predict(cv_fit.lasso[[7]], newx = Z_test, s = y, type="response"), predict(cv_fit.lasso[[8]], newx = Z_test, s = y, type="response"), 
                                      predict(cv_fit.lasso[[9]], newx = Z_test, s = y, type="response"), predict(cv_fit.lasso[[10]], newx = Z_test, s = y, type="response"))
  ZOL_train.lasso = sapply(lambdas, function(x) mean(as.integer(train[,d+1]) != apply(prob_training.lasso(x), 1, which.max)))
  ZOL_test.lasso = sapply(lambdas, function(x) mean(as.integer(test[,d+1]) != apply(prob_test.lasso(x), 1, which.max)))
  MSE_train.lasso = sapply(lambdas, function(x) mean(as.matrix(train[, (d+2):(d+11)] - prob_training.lasso(x))^2))
  MSE_test.lasso = sapply(lambdas, function(x) mean(as.matrix(test[, (d+2):(d+11)] - prob_test.lasso(x))^2))
  SE_train.lasso = sapply(lambdas, function(x) sum((train[, (d+2):(d+11)] - prob_training.lasso(x))^2))
  SE_test.lasso = sapply(lambdas, function(x) sum((test[, (d+2):(d+11)] - prob_test.lasso(x))^2))
  
  return(list(ls = list(L2_norm = L2_norm.ls,
                        ZOL_train = ZOL_train.ls, ZOL_test = ZOL_test.ls, MSE_train = MSE_train.ls, MSE_test = MSE_test.ls),
              ridge = list(lambda_min = lambda_min.ridge,
                           L2_norm = L2_norm.ridge,
                           ZOL_train = ZOL_train.ridge, ZOL_test = ZOL_test.ridge, MSE_train = MSE_train.ridge, MSE_test = MSE_test.ridge),
              lasso = list(lambda_min = lambda_min.lasso,
                           L2_norm = L2_norm.lasso,
                           ZOL_train = ZOL_train.lasso, ZOL_test = ZOL_test.lasso, MSE_train = MSE_train.lasso, MSE_test = MSE_test.lasso)))
}

reg4 = function(D, train, test){
  
  # intermediary features
  Z = sqrt(2/D)*Z_tot[, 1:D]
  Z_test = sqrt(2/D)*Z_test_tot[, 1:D]
  
  # Square loss function subject to minimum L2 norm (using Moore–Penrose inverse)
  
  alpha_ls = foreach(i = 1:10) %dopar% {
    ginv(diag(rep(sqrt(W[i]), N_training)) %*% Z) %*% diag(rep(sqrt(W[i]), N_training)) %*% train[, d+1+i]
  }
  L2_norm.ls = sum(apply(matrix(unlist(alpha_ls), nrow = D, byrow = F), 1, mean)^2)
  prob_training.ls = foreach(i = 1:10) %dopar% {
    plogis(Z%*%alpha_ls[[i]])
  }
  prob_test.ls = foreach(i = 1:10) %dopar% {
    plogis(Z_test%*%alpha_ls[[i]])
  }
  
  MSE_train.ls = mean((train[, (d+2):(d+11)] - matrix(unlist(prob_training.ls), nrow = N_training, byrow = F))^2)
  MSE_test.ls = mean((train[, (d+2):(d+11)] -  matrix(unlist(prob_test.ls), nrow = N_training, byrow = F))^2)
  ZOL_train.ls = mean(as.integer(train[,d+1]) != apply(matrix(unlist(prob_training.ls), nrow = N_training, byrow = F), 1, which.max))
  ZOL_test.ls = mean(as.integer(test[,d+1]) != apply(matrix(unlist(prob_test.ls), nrow = N_training, byrow = F), 1, which.max))
  
  return(list(ls = list(L2_norm = L2_norm.ls, ZOL_train = ZOL_train.ls, ZOL_test = ZOL_test.ls, MSE_train = MSE_train.ls, MSE_test = MSE_test.ls)))
}


#### REGRESSION - RUN #### 

## Run on cluster
D_grid = c(seq(2, 900, 50), c(1, 3, 6, 9, 10, 11, 13, 15, 20)*10^3)

reg2(D = 100, train, test)

res = list(); i = 1
for(D in D_grid) {
  res[[i]] = reg(D, train, test)
  i = i+1
}

# save(res, file = file.path("/rds/general/user/mm3218/home/res/res_reg2"))
