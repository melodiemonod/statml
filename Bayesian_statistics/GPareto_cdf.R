library("SpatialExtremes")
library(data.table)
library(ggplot2)

indir = "~/Documents/PhD/Modules/Bayesian_Methods/Project/Code"


x = seq(0, 10, 0.01); k = c(0, 0.2,0.3,0.5,0.7,1)
#df = data.table(x = rep(x, 4), k = as.vector(sapply(k, function(y) rep(y,length(x)))), 
#                GPdensity = c(dgpd(x, loc = 0, scale = 1, shape = k[1]), dgpd(x, loc = 0, scale = 1, shape = k[2]), dgpd(x, loc = 0, scale = 1, shape = k[3]), dgpd(x, loc = 0, scale = 1, shape = k[4])))
df = data.table(x = rep(x, length(k)), k = as.vector(sapply(k, function(y) rep(y,length(x)))), 
                GPdensity = c(pgpd(x, loc = 0, scale = 1, shape = k[1]), pgpd(x, loc = 0, scale = 1, shape = k[2]), pgpd(x, loc = 0, scale = 1, shape = k[3]), pgpd(x, loc = 0, scale = 1, shape = k[4]), 
                              pgpd(x, loc = 0, scale = 1, shape = k[5]), pgpd(x, loc = 0, scale = 1, shape = k[6]))) 

df$k = as.factor(df$k)

ggplot(df, aes(x = x, y=GPdensity, color = k)) +
  geom_line() +
  xlab("x") +
  ylab(expression(f[GP](x)))+
  theme_bw()+
  theme(text = element_text(size=25))
ggsave("GP_density.png", w = 10, h = 7, device = "png", path = file.path(indir, "figures"))




