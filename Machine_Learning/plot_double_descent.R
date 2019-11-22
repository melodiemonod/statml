library("gridExtra")
library("ggplot2")

#### DOUBLE DESCENT CURVE ####
# This file intend to replicate RFF results obtained in Belkin et al. 2018

#  directory
indir = "~/Documents/PhD/Modules/Machine_Learning/code"

#### PLOTS #### 

load(file.path(indir, "res1"))
df= unlist(res)
load(file.path(indir, "res2"))
df2= unlist(res)
load(file.path(indir, "res3"))
df3= unlist(res)

lambdas = c(1000, 100,  30, 10, 3, 1, 0.3, 0.1, 0.03, 0.01, 0.001, 0.0001); 
D_grid = c(seq(2, 900, 100), c(1, 2,4,5,7,10,13,15,20)*10^3)
tmp = data.frame(method = c(rep("LS", length(D_grid)), rep(paste("Ridge lambda =", round(lambdas, digits = 4)), length(D_grid)) , rep(paste("Lasso lambda =", round(lambdas, digits = 4)), length(D_grid))), 
                 D = c(D_grid, rep(as.vector(sapply(D_grid, function(x) rep(x, length(lambdas)))), 2)),
                 MSE_training_i  = c(df[grepl( "ls.MSE_train", names(df))], df[grepl( "ridge.MSE_train", names(df))], df[grepl( "lasso.MSE_train", names(df))]),
                 MSE_test_i  = c(df[grepl( "ls.MSE_test", names(df))], df[grepl( "ridge.MSE_test", names(df))], df[grepl( "lasso.MSE_test", names(df))]),
                 MSE_training_l  = c(df2[grepl( "ls.MSE_train", names(df2))], df2[grepl( "ridge.MSE_train", names(df2))], df2[grepl( "lasso.MSE_train", names(df2))]),
                 MSE_test_l  = c(df2[grepl( "ls.MSE_test", names(df2))], df2[grepl( "ridge.MSE_test", names(df2))], df2[grepl( "lasso.MSE_test", names(df2))]),
                 ZOL_training_i  = c(df[grepl( "ls.ZOL_train", names(df))], df[grepl( "ridge.ZOL_train", names(df))], df[grepl( "lasso.ZOL_train", names(df))]),
                 ZOL_test_i  = c(df[grepl( "ls.ZOL_test", names(df))], df[grepl( "ridge.ZOL_test", names(df))], df[grepl( "lasso.ZOL_test", names(df))]),
                 ZOL_training_l  = c(df2[grepl( "ls.ZOL_train", names(df2))], df2[grepl( "ridge.ZOL_train", names(df2))], df2[grepl( "lasso.ZOL_train", names(df2))]),
                 ZOL_test_l  = c(df2[grepl( "ls.MSE_test", names(df2))], df2[grepl( "ridge.ZOL_test", names(df2))], df2[grepl( "lasso.ZOL_test", names(df2))]),
                 L2_norm_i = c(df[grepl( "ls.L2_norm", names(df))], df[grepl( "ridge.L2_norm", names(df))], df[grepl( "lasso.L2_norm", names(df))]), 
                 L2_norm_l = c(df2[grepl( "ls.L2_norm", names(df2))], df2[grepl( "ridge.L2_norm", names(df2))], df2[grepl( "lasso.L2_norm", names(df2))]))

tmp2 = data.frame(method = c(rep("LS", length(D_grid))), 
                  D = c(D_grid), 
                  MSE_training  = c(df3[grepl( "ls.MSE_train", names(df3))]),
                  MSE_test  = c(df3[grepl( "ls.MSE_test", names(df3))]),
                  ZOL_training  = c(df3[grepl( "ls.ZOL_train", names(df3))]),
                  ZOL_test  = c(df3[grepl( "ls.ZOL_test", names(df3))]),
                  L2_norm = c(df3[grepl( "ls.L2_norm", names(df3))]))


p1.lasso = ggplot(tmp[grepl("Lasso", tmp$method),], aes(x = D, color = factor(method, levels = paste("Lasso lambda =", round(lambdas, digits = 4))))) +
  geom_line(aes(y = MSE_test_l))+
  geom_point(aes(y = MSE_test_l), shape = 4)+
  labs(title = "Lasso - Logit link function")+
  theme_bw() +
  ylab("Mean Squared Error Test") +
  theme(axis.title.x=element_blank(), legend.position = "none", text = element_text(size=18))+ 
  labs(color = "Method")

p2.lasso = ggplot(tmp[grepl("Lasso", tmp$method),], aes(x = D, color = factor(method, levels = paste("Lasso lambda =", round(lambdas, digits = 4))))) +
  geom_line(aes(y = L2_norm_l))+
  geom_point(aes(y = L2_norm_l), shape = 4)+
  theme_bw()+
  ylab("Norm") +
  theme(axis.title.x=element_blank(), legend.position = "none", text = element_text(size=18))+ 
  labs(color = "Method")

p3.lasso = ggplot(tmp[grepl("Lasso", tmp$method),], aes(x = D, color = factor(method, levels = paste("Lasso lambda =", round(lambdas, digits = 4))))) +
  geom_line(aes(y = MSE_training_l))+
  geom_point(aes(y = MSE_training_l), shape = 4)+
  theme_bw() +
  ylab("Mean Squared Error Train") +
  xlab("Number of Random Fourier Features (N)") + 
  labs(color = "Method")+ 
  theme(legend.position = "bottom", text = element_text(size=18))+
  guides(color=guide_legend(nrow=4,byrow=TRUE))
plot_lasso = grid.arrange(p1.lasso, p2.lasso, p3.lasso,nrow = 3, ncol =1, heights = c(4,4,6))
ggsave(plot = plot_lasso, w = 9, h = 12, device = "pdf", filename  = file.path(indir,"plot_lasso.pdf"))


p1.ridge = ggplot(tmp[grepl("Ridge", tmp$method),], aes(x = D, color = factor(method, levels = paste("Ridge lambda =", round(lambdas, digits = 4))))) +
  geom_line(aes(y = MSE_test_l))+
  geom_point(aes(y = MSE_test_l), shape = 4)+
  labs(title = "Ridge - Logit link function")+
  theme_bw() +
  ylab("Test") +
  theme(axis.title.x=element_blank(), legend.position = "none", text = element_text(size=18))+ 
  labs(color = "Method") +
  ylab("Mean Squared Error Test") 

p2.ridge = ggplot(tmp[grepl("Ridge", tmp$method),], aes(x = D, color = factor(method, levels = paste("Ridge lambda =", round(lambdas, digits = 4))))) +
  geom_line(aes(y = L2_norm_l))+
  geom_point(aes(y = L2_norm_l), shape = 4)+
  theme_bw()+
  ylab("Norm") +
  theme(axis.title.x=element_blank(), legend.position = "none", text = element_text(size=18))+ 
  labs(color = "Method")

p3.ridge =   ggplot(tmp[grepl("Ridge", tmp$method),], aes(x = D, color = factor(method, levels = paste("Ridge lambda =", round(lambdas, digits = 4))))) +
  geom_line(aes(y = MSE_training_l))+
  geom_point(aes(y = MSE_training_l), shape = 4)+
  theme_bw()+
  ylab("Mean Squared Error Train") +
  xlab("Number of Random Fourier Features (N)") + 
  labs(color = "Method")+
  theme(axis.title.x=element_blank(), legend.position = "bottom", text = element_text(size=18))+
  guides(color=guide_legend(nrow=4,byrow=TRUE))
plot_ridge = grid.arrange(p1.ridge, p2.ridge, p3.ridge, nrow = 3, ncol =1, heights = c(4,4,6))
ggsave(plot = plot_ridge, w = 9, h = 12, device = "pdf", filename  = file.path(indir,"plot_ridge.pdf"))



p1.ls = ggplot(tmp2[grepl("LS", tmp2$method),], aes(x = D, color = method)) +
  geom_line(aes(y = log(MSE_test))) +
  geom_point(aes(y = log(MSE_test)), shape = 4) +
  labs(title = "Least norm - Analytical solution") +
  theme_bw() +
  ylab("Mean squared Error Test") +
  theme(axis.title.x=element_blank(), legend.position = "none", text = element_text(size=18))

p2.ls = ggplot(tmp[grepl("LS", tmp$method),], aes(x = D, color = method)) +
  geom_line(aes(y = log(MSE_test_i))) +
  geom_point(aes(y = log(MSE_test_i)), shape = 4) +
  labs(title = expression("Least norm - Ridge approximation", lambda, " = 1e-20")) +
  theme_bw() +
  ylab("Mean squared Error Test") +
  theme(axis.title.x=element_blank(), legend.position = "none", text = element_text(size=18))

p3.ls =  ggplot(tmp2[grepl("LS", tmp2$method),], aes(x = D, color = method)) +
  geom_line(aes(y = log(L2_norm)))+
  geom_point(aes(y = log(L2_norm)), shape = 4) +
  theme_bw()+
  ylab("Norm") +
  theme(axis.title.x=element_blank(), legend.position = "none", text = element_text(size=18))

p4.ls =  ggplot(tmp[grepl("LS", tmp$method),], aes(x = D, color = method)) +
  geom_line(aes(y = log(L2_norm_i)))+
  geom_point(aes(y = log(L2_norm_i)), shape = 4) +
  theme_bw()+
  ylab("Norm") +
  theme(axis.title.x=element_blank(), legend.position = "none", text = element_text(size=18))

p5.ls = ggplot(tmp2[grepl("LS", tmp2$method),], aes(x = D, color = method)) +
  geom_line(aes(y = MSE_training))+
  geom_point(aes(y = MSE_training), shape = 4)+
  theme_bw()+
  ylab("Mean squared Error Train") +
  xlab("Number of Random Fourier Features (N)") +
  theme(legend.position = "none")

p6.ls = ggplot(tmp[grepl("LS", tmp$method),], aes(x = D, color = method)) +
  geom_line(aes(y = MSE_training_i))+
  geom_point(aes(y = MSE_training_i), shape = 4)+
  theme_bw()+
  ylab("Mean squared Error Train") +
  xlab("Number of Random Fourier Features (N)")  +
  theme(legend.position = "none", text = element_text(size=18))

plot_ls = grid.arrange(p1.ls, p2.ls, p3.ls, p4.ls, p5.ls, p6.ls, nrow = 3, ncol =2, heights = c(4, 3, 3))
ggsave(plot = plot_ls, w = 10, h = 10, device = "pdf", filename  = file.path(indir,"plot_leastnorm.pdf"))
