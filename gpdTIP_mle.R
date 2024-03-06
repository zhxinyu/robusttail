rm(list=ls())
gof_find_threshold <- function(data, bootstrap=FALSE) {
  data <- as.numeric(data)
  if (bootstrap) {
    probs <- seq(0.6, 0.9, length.out = 10)
  } else {
    probs <- seq(0.6, 0.9, length.out = 100)

  }
  thresholds <- as.numeric(quantile(data, probs = probs))
  pvalues <- c()
  valid_threshold <- c()
  for (threshold_idx in seq_along(thresholds)) {
    tryCatch({
      if (bootstrap) {
        test_result <- eva::gpdSeqTests(data=data,
                                        thresholds=thresholds[threshold_idx],
                                        method="ad",
                                        nsim=100)
      } else {
        test_result <- eva::gpdSeqTests(data=data,
                                        thresholds=thresholds[threshold_idx],
                                        method="ad")

      }
      pvalues <- append(pvalues, test_result$p.values)
      valid_threshold <- append(valid_threshold, thresholds[threshold_idx])
    }, error = function(e) {
      # print(e)
    })
  }
  return(valid_threshold[which.max(pvalues)])
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

	CI <- hHat + qnorm(c(alpha/2, 1-alpha/2))*sdHat
	output <- (1 - Fnu)*data.frame(lB = CI[1], uB = CI[2])
	}


	return(output)
}

# a = {:}

# b = {:}

# data <- c({:})

gpdTIP <- function(data, lhs, rhs, conf = .95) {

	if(lhs >= rhs) {
		stop("Left end point must be smaller than right end point!")
	}

	u <- gof_find_threshold(data)
	if (is.null(u)) {
		u <- gof_find_threshold(data, bootstrap=TRUE)
	}

	fitGPD <- tryCatch(
		QRM::fit.GPD(data,
		             threshold = u,
		             type = "ml"), 
		error = function(e) e 
		)

	Ubd <- if ("error" %in% class(fitGPD)) {
		NA
	} else{
		h <- function(xi, beta) pGPD(b -u,xi,beta) - pGPD(a -u,xi,beta)
		hGrad <- function(xi,beta) gradientpGPD(b-u,xi,beta) - gradientpGPD(a-u,xi,beta)
		asymptoticCIforGPDfit_m(fitGPD, h, hGrad, verbose = FALSE)
	}

	return(Ubd)
}