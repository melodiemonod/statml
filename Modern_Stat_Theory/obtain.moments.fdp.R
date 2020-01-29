
## Analytical moments of FDP using Benjamini-H method
indir = "~/git/multitest"

source(file = file.path(indir,"utils.R")) # Functions used in this analysis


##run in the cluster
library(doParallel)
library("data.table")
library("tidyverse")


doParallel::registerDoParallel()

## moments
# diff m 
n = 10; m = c(10,100,1000); delta = 1
moments.pdf_m = foreach(i = m) %dopar% {
  moments(m = i, pi0 = .5, F1 = function(y) F1(y, n=n, delta_random = F, delta = delta))
}

# diff n
n = c(1, 10,100, 1000); m = 10; delta = 1
moments.fdp_n = foreach(i = n) %dopar% {
  moments(m = m, pi0 = .5, F1 = function(y) F1(y, n=i, delta_random = F, delta = delta))
}
moments.fdp_n = t(matrix(ncol = length(n), nrow  = 2, round(unlist(moments.fdp_n), digits = 6)))
colnames(moments.fdp_n) = c("Expectation", "Sd")
rownames(moments.fdp_n) = n

# diff delta
n = 10; m = 10; delta = c(.1, .5,1, 3)
moments.fdp_delta = foreach(i = delta) %dopar% {
  moments(m = m, pi0 = .5, F1 = function(y) F1(y, n=n, delta_random = F, delta = i))
}
moments.fdp_delta = t(matrix(ncol = length(delta), nrow  = 2, round(unlist(moments.fdp_delta), digits = 6)))
colnames(moments.fdp_delta) = c("Expectation", "Variance")
rownames(moments.fdp_delta) = delta

#diff pi
n = 10; m = 10; delta = 1; pi0 = c(0,0.2,0.5,0.8,1)
moments.fdp_pi = foreach(i = pi0) %dopar% {
  moments(m = m, pi0 = i, F1 = function(y) F1(y, n=n, delta_random = F, delta = delta))
}
moments.fdp_pi = t(matrix(ncol = length(pi0), nrow  = 2, round(unlist(moments.fdp_pi), digits = 6)))
colnames(moments.fdp_pi) = c("Expectation", "Variance")
rownames(moments.fdp_pi) = delta

