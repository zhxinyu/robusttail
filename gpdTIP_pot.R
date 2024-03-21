rm(list=ls())

source("common_utils.R")

MEplotFindThresh <- function (data) {
	data <- as.numeric(data)
	plt<-POT::mrlplot(data, u.range = c(quantile(data, probs = 0.6), quantile(data, probs = 0.995)), nt = 200)
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

gpdTIP <- function(data, lhs, rhs, conf = .95) {

	if(lhs >= rhs) {
		stop("Left end point must be smaller than right end point!")
	}

	u <- MEplotFindThresh(data)$thresh

	fitGPD <- tryCatch(
		QRM::fit.GPD(data,
		             threshold = u,
		             type = "ml"), 
		error = function(e) e 
		)

	Ubd <- if ("error" %in% class(fitGPD)) {
		NA
	} else{
		h <- function(xi, beta) QRM::pGPD(rhs -u,xi,beta) - QRM::pGPD(lhs -u,xi,beta)
		hGrad <- function(xi,beta) gradientpGPD(rhs-u,xi,beta) - gradientpGPD(lhs-u,xi,beta)
		asymptoticCIforGPDfit_m(fitGPD, h, hGrad, verbose = FALSE)
	}

	return(list(CI=Ubd))
}


# lhs <- 3.3174483005106072

# rhs <- 3.9397192883112084

# data <-  c(t(read.csv("./large_data/gamma/default/randomseed=20220222.csv", header=FALSE)))

# out <- gpdTIP(data, lhs, rhs, conf=0.95)
# print(out)
# if ("error" %in% class(out)) {
#   print(out)
# }

# bbd <- if ("error" %in% class(out)) NA else{
# 	out$CI
# }
# bbd 
