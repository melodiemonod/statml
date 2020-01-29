
## Analytical cdf of FDP using Benjamini-H method
indir = "~/git/multitest"
outdir = "~"
source(file = file.path(indir,"utils.R")) # Functions used in this analysis


##run in the cluster
library(doParallel)
library("data.table")
library("tidyverse")


doParallel::registerDoParallel()

x = seq(0,1,.01)

# move m
n = 10; m = c(10,100,1000); delta = 1
cdf.fdp_m = foreach(i = m) %dopar% {
  sapply(x, function(x) pfdp(x, m = i, pi0 = .5, F1 = function(y) F1(y, n=n, delta_random = F, delta = delta)))
}
saveRDS(cdf.fdp_m, file.path(indir, "results", "cdf.fdp_m.rds"))


# move n
n = c(1, 10,100, 1000); m = 10; delta = 1
cdf.fdp_n = foreach(i = n) %dopar% {
  sapply(x, function(x) pfdp(x, m = m, pi0 = .5, F1 = function(y) F1(y,n=i, delta_random = F, delta = delta)))
}
saveRDS(cdf.fdp_n, file.path(indir, "results", "cdf.fdp_n.rds"))

# move delta
n = 10; m = 10; delta = c(.1, .5,1, 3)
cdf.fdp_delta = foreach(i = delta) %dopar% {
  sapply(x, function(x) pfdp(x, m = m, pi0 = .5, F1 = function(y) F1(y,n=n, delta_random = F, delta = i)))
}
saveRDS(cdf.fdp_delta, file.path(indir, "results", "cdf.fdp_delta.rds"))

# move pi0
n = 10; m = 10; delta = 1; pi0 = c(0,0.2,0.5,0.8,1)
cdf.fdp_pi = foreach(i = pi0) %dopar% {
  sapply(x, function(x) pfdp(x, m = m, pi0 = i, F1 = function(y) F1(y, n=n, delta_random = F, delta = delta)))
}
saveRDS(cdf.fdp_pi, file.path(indir, "results", "cdf.fdp_pi.rds"))
