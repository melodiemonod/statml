library("data.table") 
library(rstan) ### stan is used for VI
library(loo)  ### loo is used for PSIS

# Parts of this code has been copy-pasted from R code for "Yes, but Did It Work: Evaluating Variational Inference" in https://github.com/yao-yl/Evaluating-Variational-Inference
# Copyright (c) 2018 Yuling Yao


# change 
setwd("/rds/general/user/mm3218/home/projects/CDT_modules/bayesian_stats")
indir = "/rds/general/user/mm3218/home/projects/CDT_modules/bayesian_stats"

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())


## SIMULATE DATA

n = 5; sigma = 1.2
x = matrix(nrow = n, ncol = 2, c(rep(1, n), 0:(n-1)))
set.seed(817952)
cov0 = diag(rep(10^2,2)); mean0 = cbind(rep(0,2))
alpha = rnorm(1,0,10); beta = rnorm(1,0,10)
y = cbind(rnorm(n, alpha + x[,2]*beta, sigma))


## EXACT POSTERIOR DISTRIBUTION

cov_posterior = solve(solve(cov0) + t(x)%*%x/(sigma^2))
mean_posterior = cov_posterior%*%(solve(cov0)%*%mean0 + t(x)%*%y/(sigma^2))


## BAYESIAN LINEAR REGRESSION

stan_code='
data {
int <lower=0> n;
int <lower=0> D;
vector [n] x ;
vector [n] y;
vector [D] sd_prior ;
vector [D] mean_prior ;
}
parameters {
real alpha;
real beta;
}
model {
y ~ normal(alpha + x * beta, 1.2);
alpha ~ normal(mean_prior[1], sd_prior[1]);
beta ~ normal(mean_prior[2], sd_prior[2]);
}
generated quantities{
real  log_density;
log_density=normal_lpdf(y |alpha + x * beta, 1.2)+normal_lpdf(alpha|mean_prior[1], sd_prior[1])+normal_lpdf(beta|mean_prior[2], sd_prior[2]) ; 
}
'

m=stan_model(model_code = stan_code)   
ip_weighted_average=function(lw, x){
  ip_weights=exp(lw-min(lw))
  return(  t(ip_weights)%*%x /sum(ip_weights))
}


## FIND k AND PMSE

tol = 0.00001 # choose relative tolerance for VI.
sim_N=100   # repeat 100 simulations
M = 5 # true posterior and 4 priors
# ceate a list with the priors
prior = list(prior1 = list(mean_prior = c(0,0), sd_prior = c(10,10)),
             prior2 = list(mean_prior = c(0,0), sd_prior = c(10, 2)), 
             prior3 = list(mean_prior = c(0,0), sd_prior = c(10, 100)),
             prior4 = list(mean_prior = c(0,100), sd_prior = c(10,10)))

K_hat=matrix(NA,sim_N,I); RMSE = matrix(NA,sim_N,(I-1))
vi_parameter_mean = vi_parameter_sd = array(NA,c(sim_N,(I-1), 2))

set.seed(1000)
for(i in 1:M){
  for(sim_n in 1:sim_N){  
    if(i < M){
      # Sample using VI
      fit_vb=vb(m, data=list(x=x[,2],y=as.vector(y), D=2,n=n, sd_prior = prior[[i]]$sd_prior, mean_prior = prior[[i]]$mean_prior), 
                iter=5e5,output_samples=1e5,tol_rel_obj=tol,eta=0.05,adapt_engaged=F) 
      vb_samples=extract(fit_vb)
      parameter_sample=cbind(vb_samples$alpha, vb_samples$beta)
      vi_parameter_mean[sim_n,i,]=apply(parameter_sample, 2, mean)
      vi_parameter_sd[sim_n,i,]=apply(parameter_sample, 2, sd)
      normal_likelihood=function(parameter_sample){
        one_data_normal_likelihood=function(vec){
          return(sum(dnorm(vec,mean=vi_parameter_mean[sim_n,i,],sd=vi_parameter_sd[sim_n,i,],  log=T)))
        }
        return(apply(parameter_sample, 1, one_data_normal_likelihood))
      }
      # compule log weights
      lp_vi=normal_likelihood(parameter_sample)
      lp_target=vb_samples$log_density
      
    } else{
      # Sample using the true posterior
      parameter_sample=cbind(rnorm(1e5, mean_posterior[1], sqrt(cov_posterior[1,1])), rnorm(1e5, mean_posterior[2], sqrt(cov_posterior[2,2])))
      normal_likelihood_true=function(parameter_sample){
        one_data_normal_likelihood=function(vec){
          return(sum(dnorm(vec[1], mean_posterior[1], sqrt(cov_posterior[1,1]), log = T) + dnorm(vec[2], mean_posterior[2], sqrt(cov_posterior[2,2]), log = T)))
        }
        return( apply(parameter_sample, 1, one_data_normal_likelihood))
      }
      normal_model=function(parameter_sample){
        one_data_normal_likelihood=function(vec){
          return( sum( dnorm(y,mean=vec[1] + vec[2]* x[,2] ,sd=1.2,  log=T)) + dnorm(vec[1], mean0[1], sqrt(cov0[1,1]), log = T) + dnorm(vec[2], mean0[2], sqrt(cov0[2,2]), log = T))
        }
        return( apply(parameter_sample, 1, one_data_normal_likelihood))
      }
      
      # compule log weights
      lp_vi=normal_likelihood_true(parameter_sample)
      lp_target=normal_model(parameter_sample)
    }
    ip_ratio=lp_target-lp_vi
    
    # fit weights to a Generalized Pareto with loo
    ok=complete.cases(ip_ratio)
    joint_diagnoistics=psis(log_ratios=as.vector(ip_ratio[ok]))
    K_hat[sim_n,i]=joint_diagnoistics$diagnostics$pareto_k
    
    if(i < I){
      RMSE[sim_n,i]=sqrt(mean((vi_parameter_mean[sim_n,i,2]-mean_posterior[2])^2))
    }
    
    print(paste("=======================    i=",i,"   ========================"))
    print(paste("=======================iter",sim_n,"========================"))
  }
}

save(K_hat, RMSE, EDS,file=file.path(indir,"linear_model.RData"))



## PLOTS

load(file.path(indir,"linear_model.RData"))

## khat against prior ##
y.median = apply(K_hat, 2, median)
y.PI 	<- apply(K_hat, 2, function(x) quantile(x, prob=c(0.025,0.975)))
y.PI	<- rbind(y.median, y.PI)
rownames(y.PI)	<-  c('predicted_median','predicted_l95','predicted_u95')
y.PI	<- as.data.frame(t(y.PI))
ggplot(y.PI[1:4,], aes(x=1:4, ymin=predicted_l95, ymax=predicted_u95)) +
  geom_point(aes(y=predicted_median), size = 2) + 
  geom_errorbar(size =1) +
  theme_bw() +		
  geom_hline(yintercept = 0.7, color = "red", size=1) +
  xlab("Prior") + 
  ylab(expression(hat(k))) +
  theme(text = element_text(size=25))+ 
  geom_rect(aes(ymin=y.PI[5,2], ymax=y.PI[5,3], xmin=-Inf, xmax=Inf), alpha =0.2) 
ggsave("CI_k.png", w = 5, h = 6, device = "png", path = file.path(indir, "figures"))


## khat against RMSE ##
df = data.frame(RMSE = as.vector(RMSE), K_hat = as.vector(K_hat[,1:(M-1)]), Prior = as.vector(sapply(1:(M-1), function(x) rep(x, sim_N))), EDS = as.vector(EDS))
df$Prior = as.factor(df$Prior)
ggplot(df, aes(x = K_hat, y = RMSE, color = Prior)) +
  geom_line(size = 1)+
  xlab(expression(hat(k))) +
  ylab("RMSE")+
  theme_bw()+
  theme(text = element_text(size=25))
ggsave("RMSE_k.png", w = 7, h = 6, device = "png", path = file.path(indir, "figures"))


## True posterior compared to VI posteriors  ##
x_beta = seq(10.5,14,0.01); x_alpha = seq(-2,7,length.out = length(x_beta))
dff = data.frame(mean_all_alpha = c(mean_posterior[1], apply(vi_parameter_mean[,,1], 2, median)), mean_all_beta = c(mean_posterior[2], apply(vi_parameter_mean[,,2], 2, median)), 
                 sd_all_alpha = c(sqrt(diag(cov_posterior))[1], apply(vi_parameter_sd[,,1], 2, median)), sd_all_beta = c(sqrt(diag(cov_posterior))[2], apply(vi_parameter_sd[,,2], 2, median)), 
                 Posterior = c("True", paste0("Under prior", 1:(I-1))))
df2 =   data.frame(x_alpha = rep(x_alpha, I), x_beta = rep(x_beta, M), Posterior = as.vector(sapply(c("True", paste0("Under prior", 1:(M-1))), function(x) rep(x, length(x_beta)))))
dff = merge(dff, df2, by = "Posterior")
dff = as.data.table(dff)
dff[, den_posterior_alpha := dnorm(x_alpha, mean_all_alpha, sd_all_alpha)]
dff[, den_posterior_beta := dnorm(x_beta, mean_all_beta, sd_all_beta)]

# beta
ggplot(dff, aes(x = x_beta, y = den_posterior_beta, color = Posterior, linetype = Posterior)) +
  geom_line(size = 1)+
  xlab(expression(beta)) +
  ylab("density")+
  theme_bw()+
  theme(text = element_text(size=25))
ggsave("VI_density.png", w = 8, h = 6, device = "png", path = file.path(indir, "figures"))

# alpha
ggplot(dff, aes(x = x_alpha, y = den_posterior_alpha, color = Distribution)) +
  geom_line()+
  xlab(expression(alpha)) +
  ylab(expression(hat(pi(alpha))))+
  theme_bw()+
  theme(text = element_text(size=18))
