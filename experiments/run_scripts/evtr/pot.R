library(QRM)
library(POT)
MEplotFindThresh <- function (data) {
  data <- as.numeric(data)
  plt<-mrlplot(data, u.range = c(quantile(data, probs = 0.6), quantile(data, probs = 0.995)), nt = 200)
  x <- plt$x
  y <- plt$y[,'mrl']
  len <- length(x)
  #Initiation for Loop
  corr <- c()
  #Loop for linearity
  range_corr <- 1:len
  for (i in range_corr) {
    if (1-sum(data>x[i])/ length(data) >0.9){
      break
    }
    corr[i] <- cor(x[i:len],y[i:len])
  }
  
  u <- x[as.numeric(which.max(abs(corr)))]
  q_u <- 1-sum(data>u)/ length(data)
  return(list("thresh"=u, "perc.thresh"=q_u))
}

gradientpGPD <- function(x,xi,beta,eps = 1e-12)
{
  gradient <- matrix(0, ncol = length(x),nrow = 2, dimnames = list(c("xi","beta")))
  
  # Check with eps rather than 0 for numerical stability
  if (abs(xi)>= eps){
    
    supp <- 0 < (1+xi*x/beta)
    xs <- x[supp]
    
    gradient[1,supp] <- (-1/xi^2*log(1+xi*xs/beta) + xs/(xi*(beta + xi*xs)))*(1-pGPD(xs,xi,beta))
    gradient[2,] <- -x/beta*dGPD(x,xi,beta) # Partial Beta
  }
  
  # In this case, the GPD fit is numerically an exponential distribution
  if (abs(xi) < eps){
    gradient[1,] <- 0
    gradient[2,] <- x/beta*dexp(x,1/beta)
  }
  
  output <- gradient
  return(output)
}

asymptoticCIforGPDfit_m <- function(fitGPD,h,hGrad, alpha = 0.05,verbose = TRUE)
{
  
  # Getting the parameters with the library ismev
  xi   <- as.numeric(fitGPD$par.ests[1])
  beta <- as.numeric(fitGPD$par.ests[2])
  tmp     <- fitGPD$varcov
  
  Fnu <- fitGPD$p.less.thresh
  nexc <- fitGPD$n.exceed
  
  Sigma   <- matrix( (1+xi)*c( (1+xi),beta,beta,2*beta^2),ncol =2,nrow = 2)
  
  # Point estimate of h for the given estimation of xi and beta
  hHat <- h(xi,beta)
  
  if (xi < -1)   output <- data.frame(lB = NA, hHat = NA, uB = NA)
  if (xi >= -1)
  {
    
    if (nexc  < 30 && verbose ) print(paste("Unreliable Delta-Method for u =" ,u ,". Nexc < 30",sep=""))
    if (xi< -1/2 && verbose)  print(paste("Unreliable Delta-Method for u =" ,u ,". xi < -1/2",sep=""))
    
    gradient  <- hGrad(xi,beta)
    sdHat <- sqrt(t(gradient)%*%Sigma%*%gradient/nexc)
    
    CI <- hHat + qnorm(1-alpha)*sdHat
    output <- (1 - Fnu)*data.frame(uB = CI)
  }
  
  
  return(output)
}

a = qnorm(0.99)
b = qnorm(0.995)
data <- rnorm(500)
u <- MEplotFindThresh(data)$thresh
perc_u <- MEplotFindThresh(data)$perc.thresh
fitGPD <- tryCatch(
QRM::fit.GPD(data,threshold = u,type = "ml"), 
error = function(e)e)
Ubd <- if ("error" %in% class(fitGPD)) NA else{
h <- function(xi, beta) QRM::pGPD(b -u,xi,beta) - QRM::pGPD(a -u,xi,beta)
hGrad <- function(xi,beta) gradientpGPD(b-u,xi,beta) - gradientpGPD(a-u,xi,beta)
asymptoticCIforGPDfit_m(fitGPD,h,hGrad,verbose = FALSE)
}