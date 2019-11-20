
library("kernlab")

svmfit = ksvm(y = as.factor(train[, (d+1)]), x = train[, -(d+1)], kernel = "rbfdot", type = "C-svc", scale = F)

L2_norm.svm = sqrt(sum(unlist(alpha(svmfit))^2))
mean(as.factor(train[, (d+1)]) == fitted(svmfit))
mean((as.integer(train[, (d+1)]) - as.integer(fitted(svmfit)))^2)
ZOL_training.svm =  mean(as.integer(train[,d+1]) == round(predict(svmfit, newx = train[, -(d+1)]), digits = 0))
ZOL_test.svm =  mean(as.integer(test[,d+1]) == round(predict(svmfit, newx = test[, -(d+1)]), digits = 0))
MSE_train.svm = log(mean((as.integer(train[,d+1]) - predict(svmfit, newx = train[, -(d+1)]))^2)))
MSE_test.svm = log(mean((as.integer(test[,d+1]) - predict(svmfit, newx = test[, -(d+1)]))^2)))
