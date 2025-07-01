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
	## beta is scale, xi is -shape
	Ubd <- if ("error" %in% class(fitGPD)) {
		NA
	} else{
		h <- function(xi, beta) {
			if (rhs == Inf) {
				return(1-QRM::pGPD(lhs -u,xi,beta))
			}
			return(QRM::pGPD(rhs -u,xi,beta) - QRM::pGPD(lhs -u,xi,beta))
		}
		hGrad <- function(xi,beta) {
			if (rhs == Inf) {
				return(-gradientpGPD(lhs -u,xi,beta))
			}
			return(gradientpGPD(rhs-u,xi,beta) - gradientpGPD(lhs-u,xi,beta))
		}
		asymptoticCIforGPDfit_m(fitGPD, h, hGrad, alpha = 1 - conf, verbose = FALSE)
	}

	return(list(CI=Ubd))
}


# lhs <- 3.3174483005106072

# rhs <- 3.9397192883112084

# data <-  c(t(read.csv("./large_data/gamma/default/randomseed=20220222.csv", header=FALSE)))

# out <- gpdTIP(data, lhs, rhs, conf=0.95)
# print(out)