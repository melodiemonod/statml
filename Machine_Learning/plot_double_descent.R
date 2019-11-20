library("gridExtra")
library("ggplot2")

#### DOUBLE DESCENT CURVE ####
# This file intend to replicate RFF results obtained in Belkin et al. 2018

#  directory
indir = "~/Documents/PhD/Modules/Machine_learning/code"

#### PLOTS #### 

load(file.path(indir, "res"))

lambdas = c(1000, 100,  30, 10, 3, 1, 0.3, 0.1, 0.03, 0.01, 0.001, 0.0001); lambda_ls = 1e-14
D_grid = c(seq(2, 900, 100), c(1, 2,4,5,7,10,13,15,20)*10^3)
df= unlist(res)
tmp = data.frame(method = c(rep("LS", length(D_grid)), rep(paste("Ridge lambda =", round(lambdas, digits = 4)), length(D_grid)) , rep(paste("Lasso lambda =", round(lambdas, digits = 4)), length(D_grid))), 
                 D = c(D_grid, rep(as.vector(sapply(D_grid, function(x) rep(x, length(lambdas)))), 2)), 
                 MSE_training  = c(df[grepl( "ls.MSE_training", names(df))], df[grepl( "ridge.MSE_training", names(df))], df[grepl( "lasso.MSE_training", names(df))]),
                 MSE_test  = c(df[grepl( "ls.MSE_test", names(df))], df[grepl( "ridge.MSE_test", names(df))], df[grepl( "lasso.MSE_test", names(df))]),
                 ZOL_test  = c(df[grepl( "ls.ZOL_training", names(df))], df[grepl( "ridge.ZOL_training", names(df))], df[grepl( "lasso.ZOL_training", names(df))]),
                 ZOL_training  = c(df[grepl( "ls.ZOL_test", names(df))], df[grepl( "ridge.ZOL_test", names(df))], df[grepl( "lasso.ZOL_test", names(df))]),
                 L2_norm = sqrt(c(df[grepl( "ls.L2_norm", names(df))], df[grepl( "ridge.L2_norm", names(df))], df[grepl( "lasso.L2_norm", names(df))])),
                 lambda.opt = c(rep(lambda_ls, length(D_grid)), as.vector(sapply(df[grepl( "ridge.lambda_min", names(df))], function(x) rep(x, length(lambdas)))), 
                                as.vector(sapply(df[grepl( "lasso.lambda_min", names(df))], function(x) rep(x, length(lambdas))))))

p1.ridge = ggplot(tmp[grepl("Ridge", tmp$method),], aes(x = D, color = factor(method, levels = paste("Ridge lambda =", round(lambdas, digits = 4))))) +
  geom_line(aes(y = ZOL_test)) +
  geom_point(aes(y = ZOL_test), shape = 4) +
  labs(title = "Zero One Loss - Ridge") +
  theme_bw() +
  ylab("Test (%)") +
  theme(axis.title.x=element_blank())+ 
  labs(color = "Method")

p2.ridge = ggplot(tmp[grepl("Ridge", tmp$method),], aes(x = D, color = factor(method, levels = paste("Ridge lambda =", round(lambdas, digits = 4))))) +
  geom_line(aes(y = MSE_test))+
  geom_point(aes(y = MSE_test), shape = 4)+
  labs(title = "Mean Squared Error - Ridge")+
  theme_bw() +
  ylab("Test") +
  theme(axis.title.x=element_blank())+ 
  labs(color = "Method")

p3.ridge =  ggplot(tmp[grepl("Ridge", tmp$method),], aes(x = D, color = factor(method, levels = paste("Ridge lambda =", round(lambdas, digits = 4))))) +
  geom_line(aes(y = L2_norm))+
  geom_point(aes(y = L2_norm), shape = 4) +
  theme_bw()+
  ylab("Norm") +
  theme(axis.title.x=element_blank())+ 
  labs(color = "Method")

p4.ridge = ggplot(tmp[grepl("Ridge", tmp$method),], aes(x = D, color = factor(method, levels = paste("Ridge lambda =", round(lambdas, digits = 4))))) +
  geom_line(aes(y = L2_norm))+
  geom_point(aes(y = L2_norm), shape = 4)+
  theme_bw()+
  ylab("Norm") +
  theme(axis.title.x=element_blank())+ 
  labs(color = "Method")

p5.ridge =   ggplot(tmp[grepl("Ridge", tmp$method),], aes(x = D, color = factor(method, levels = paste("Ridge lambda =", round(lambdas, digits = 4))))) +
  geom_line(aes(y = ZOL_training))+
  geom_point(aes(y = ZOL_training), shape = 4)+
  theme_bw()+
  ylab("Train (%)") +
  xlab("Number of Random Fourier Features (N)") + 
  labs(color = "Method")

p6.ridge = ggplot(tmp[grepl("Ridge", tmp$method),], aes(x = D, color = factor(method, levels = paste("Ridge lambda =", round(lambdas, digits = 4))))) +
  geom_line(aes(y = MSE_training))+
  geom_point(aes(y = MSE_training), shape = 4)+
  theme_bw() +
  ylab("Train") +
  xlab("Number of Random Fourier Features (N)") + 
  labs(color = "Method")

plot_ridge = grid.arrange(p1.ridge, p2.ridge, p3.ridge, p4.ridge, p5.ridge, p6.ridge, nrow = 3, ncol =2, heights = c(4, 3, 3))
ggsave(plot = plot_ridge, w = 13, h = 10, device = "pdf", filename  = file.path(indir,"plot_ridge.pdf"))


p1.lasso = ggplot(tmp[grepl("Lasso", tmp$method),], aes(x = D, color = factor(method, levels = paste("Lasso lambda =", round(lambdas, digits = 4))))) +
  geom_line(aes(y = ZOL_test)) +
  geom_point(aes(y = ZOL_test), shape = 4) +
  labs(title = "Zero One Loss - Lasso") +
  theme_bw() +
  ylab("Test (%)") +
  theme(axis.title.x=element_blank())+ 
  labs(color = "Method")

p2.lasso = ggplot(tmp[grepl("Lasso", tmp$method),], aes(x = D, color = factor(method, levels = paste("Lasso lambda =", round(lambdas, digits = 4))))) +
  geom_line(aes(y = MSE_test))+
  geom_point(aes(y = MSE_test), shape = 4)+
  labs(title = "Mean Squared Error - Lasso")+
  theme_bw() +
  ylab("Test") +
  theme(axis.title.x=element_blank())+ 
  labs(color = "Method")

p3.lasso =  ggplot(tmp[grepl("Lasso", tmp$method),], aes(x = D, color = factor(method, levels = paste("Lasso lambda =", round(lambdas, digits = 4))))) +
  geom_line(aes(y = L2_norm))+
  geom_point(aes(y = L2_norm), shape = 4) +
  theme_bw()+
  ylab("Norm") +
  theme(axis.title.x=element_blank())+ 
  labs(color = "Method")

p4.lasso = ggplot(tmp[grepl("Lasso", tmp$method),], aes(x = D, color = factor(method, levels = paste("Lasso lambda =", round(lambdas, digits = 4))))) +
  geom_line(aes(y = L2_norm))+
  geom_point(aes(y = L2_norm), shape = 4)+
  theme_bw()+
  ylab("Norm") +
  theme(axis.title.x=element_blank())+ 
  labs(color = "Method")

p5.lasso =   ggplot(tmp[grepl("Lasso", tmp$method),], aes(x = D, color = factor(method, levels = paste("Lasso lambda =", round(lambdas, digits = 4))))) +
  geom_line(aes(y = ZOL_training))+
  geom_point(aes(y = ZOL_training), shape = 4)+
  theme_bw()+
  ylab("Train (%)") +
  xlab("Number of Random Fourier Features (N)") + 
  labs(color = "Method")

p6.lasso = ggplot(tmp[grepl("Lasso", tmp$method),], aes(x = D, color = factor(method, levels = paste("Lasso lambda =", round(lambdas, digits = 4))))) +
  geom_line(aes(y = MSE_training))+
  geom_point(aes(y = MSE_training), shape = 4)+
  theme_bw() +
  ylab("Train") +
  xlab("Number of Random Fourier Features (N)") + 
  labs(color = "Method")

plot_lasso = grid.arrange(p1.lasso, p2.lasso, p3.lasso, p4.lasso, p5.lasso, p6.lasso, nrow = 3, ncol =2, heights = c(4, 3, 3))
ggsave(plot = plot_lasso, w = 13, h = 10, device = "pdf", filename  = file.path(indir,"plot_lasso.pdf"))



p1.ls = ggplot(tmp[grepl("LS", tmp$method),], aes(x = D, color = method)) +
  geom_line(aes(y = ZOL_test)) +
  geom_point(aes(y = ZOL_test), shape = 4) +
  labs(title = "Zero One Loss - Least norm") +
  theme_bw() +
  ylab("Test (%)") +
  theme(axis.title.x=element_blank())

p2.ls = ggplot(tmp[grepl("LS", tmp$method),], aes(x = D, color = method)) +
  geom_line(aes(y = MSE_test))+
  geom_point(aes(y = MSE_test), shape = 4)+
  labs(title = "Mean Squared Error - Least norm")+
  theme_bw() +
  ylab("Test") +
  theme(axis.title.x=element_blank())

p3.ls =  ggplot(tmp[grepl("LS", tmp$method),], aes(x = D, color = method)) +
  geom_line(aes(y = L2_norm))+
  geom_point(aes(y = L2_norm), shape = 4) +
  theme_bw()+
  ylab("Norm") +
  theme(axis.title.x=element_blank())

p4.ls = ggplot(tmp[grepl("LS", tmp$method),], aes(x = D, color = method)) +
  geom_line(aes(y = L2_norm))+
  geom_point(aes(y = L2_norm), shape = 4)+
  theme_bw()+
  ylab("Norm") +
  theme(axis.title.x=element_blank())

p5.ls =   ggplot(tmp[grepl("LS", tmp$method),], aes(x = D, color = method)) +
  geom_line(aes(y = ZOL_training))+
  geom_point(aes(y = ZOL_training), shape = 4)+
  theme_bw()+
  ylab("Train (%)") +
  xlab("Number of Random Fourier Features (N)") 

p6.ls = ggplot(tmp[grepl("LS", tmp$method),], aes(x = D, color = method)) +
  geom_line(aes(y = MSE_training))+
  geom_point(aes(y = MSE_training), shape = 4)+
  theme_bw() +
  ylab("Train") +
  xlab("Number of Random Fourier Features (N)") 

plot_ls = grid.arrange(p1.ls, p2.ls, p3.ls, p4.ls, p5.ls, p6.ls, nrow = 3, ncol =2, heights = c(4, 3, 3))
ggsave(plot = plot_ls, w = 13, h = 10, device = "pdf", filename  = file.path(indir,"plot_leastnorm.pdf"))