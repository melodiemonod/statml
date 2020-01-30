F1 = function(x, n, delta=NULL, delta_random = F, theta = NULL, tau = NULL)
{
  #distribution of the p values under the alternative
  
  ifelse(delta_random == F, return(1 - pnorm(qnorm(x) - sqrt(n)*delta)),
         return(1 - pnorm((qnorm(x) - sqrt(n)*theta)/(tau^2*n + 1)^(1/2))))
}


pfdp = function(x, m, pi0, F1)
{
  # Analytical cdf distribution of FDP using Benjamini-H method
  F0 = function(x) punif(x, 0, 1); t = 0.05*1:m/m
  G = function(tk) pi0*F0(tk) + (1-pi0)*F1(tk)
  bolshev.rec = function(k, t){
    Psi <- rep(0, times = length(t) + 1)	
    Psi[1] <- 1 
    summand <- function(s) choose(k,s) * Psi[s+1] * (1 - t[s+1])^(k-s)
    for(k in 1:length(t))
      Psi[k+1] <- 1 - sum( sapply(0:(k-1), summand))
    return(Psi[length(t)+1])
  }
  
  part1 = function(k,j) choose(k,j) * (pi0*F0(t[k])/G(t[k]))^j * ((1 - pi0)*F1(t[k])/G(t[k]))^(k-j)
  part2 = function(k) choose(m,k) * G(ifelse(k ==0, 1, (t[k])))^k * ifelse(k == m, 1, bolshev.rec(m-k,1 - G(t[m:(k+1)])))
  part2res = as.vector(unlist(sapply(0:(m-1), function(k) part2(k))))
  part2res[m] =  choose(m,m) * (G(t[m]))^m *1
  prob = function(k,j) part1(k,j)*part2res[k]
  df = data.table(k = unlist(sapply(0:m, function(k) rep(k, 1+floor(x*k)))),
                  j = unlist(sapply(0:m, function(k) 0:floor(x*k)))) %>%
    mutate(prob := mcmapply(prob, k, j))
  sum(unlist(df$prob))
}

power = function(F1)
{
  # Analytical power of multiple testing using Benjamini-H method
  F0 = function(x) punif(x, 0, 1); t = 0.05*1:m/m
  G = function(tk) pi0*F0(tk) + (1-pi0)*F1(tk)
  part1 = function(k) log(F1(t[k]))
  part2 = function(t,k) lchoose(m-1,k-1) + (k-1)*log(t[k]) + log(bolshev.rec(m-k, 1 - t[m:(k+1)]))
  lprob = function(k) sum(part1(k)+part2(G(t[1:m]), k))
  df = data.table(k = 1:m) %>%
    mutate(lprob := mcmapply(lprob, k))
  sum(exp(df$lprob))
}

moments = function(m, F1, pi0){
  F0 = function(x) punif(x, 0, 1); t = 0.05*1:m/m
  G = function(tk) pi0*F0(tk) + (1-pi0)*F1(tk)
  bolshev.rec = function(k, t){
    Psi <- rep(0, times = length(t) + 1)	
    Psi[1] <- 1
    summand <- function(s) choose(k,s) * Psi[s+1] * (1 - t[s+1])^(k-s)
    for(k in 1:length(t))
      Psi[k+1] <- 1 - sum( sapply(0:(k-1), summand))
    return(Psi[length(t)+1])
    }
  
  ## first moment
  part1 = choose(m, m-1) * pi0 
  part2 = function(k)  F0(ifelse(k==0, 0, t[k]))/k 
  part3 = function(k) choose(m,k) * G(ifelse(k ==0, 1, (t[k])))^k * ifelse(k == m, 1, bolshev.rec(m-k,1 - G(t[m:(k+1)])))
  comp = function(k) part2(k)*part3(k-1)
  df = data.table(k = 1:m) %>%
    mutate(value := mcmapply(comp, k))
  firstmoment = part1*sum(unlist(df$value))
  
  ## second moment
  # l =1
  part1a = choose(m, m-1) * pi0 
  part2a = function(k)  F0(t[k])/(k^2) 
  compa = function(k) part2a(k)*part3( k-1)
  # l = 2
  part1b = choose(m, m-2) * pi0^2 
  part2b = function(k)  (F0(t[k])^2)/(k^2) 
  compb = function(k) part2b(k)*part3( k-2)
  df = data.table(k = 1:m) %>%
    mutate(valuea := mcmapply(compa, k), 
           valueb := mcmapply(compb, k))
  secondmoment = part1a*sum(unlist(df$valuea))+part1b*sum(unlist(df$valueb)[-1])
  sd = sqrt(secondmoment - firstmoment^2)
  
 return(c(firstmoment, sd))
}
