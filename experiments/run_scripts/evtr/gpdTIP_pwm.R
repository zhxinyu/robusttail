# rm(list=ls())

# source("common_utils.R")

asymptoticCIforGPD_delta <- function(fitGPD, h, hGrad, alpha=0.05, verbose = TRUE){
	# Delta method

    # Unpack the list using a loop
    for (name in names(fitGPD)) {
        assign(name, fitGPD[[name]])
    }

	# Point estimate of h for the given estimation of shape and scale
	hHat_pwm <- h(shape=shape_pwm, scale=scale_pwm)

	if (shape_pwm >= 0.5) {
		output <- c(prob_over_threshold*hHat_pwm, prob_over_threshold*hHat_pwm)
	} else {
		# if (nexc  < 30 && verbose ) print(paste("Unreliable Delta-Method for u =" ,u ,". Nexc < 30",sep=""))
		# if (shape< -1/2 && verbose)  print(paste("Unreliable Delta-Method for u =" ,u ,". shape < -1/2",sep=""))
		gradient_pwm  <- hGrad(shape=shape_pwm,scale=scale_pwm)
    	sdHat_pwm <- sqrt(t(gradient_pwm)%*%Sigma_pwm%*%gradient_pwm)
        if (is.na(sdHat_pwm)) {
            sdHat_pwm <- 0
        }
		CI <- hHat_pwm + qnorm(c(alpha/2, 1-alpha/2))*c(sdHat_pwm)
		output <- prob_over_threshold*c(CI[1], CI[2])
	}

	return(output)
}

gpdTIP <- function(data, lhs, rhs, conf = .95, u=NULL) {

	if(lhs >= rhs) {
		stop("Left end point must be smaller than right end point!")
	}

	if (is.null(u)) {
        u <- gof_find_threshold(data)
        if (is.null(u)) {
            u <- gof_find_threshold(data, bootstrap=TRUE)
        }
    }
    
	prob_over_threshold <- sum(data>u)/length(data)
    data <- sort(data[data>u]-u)

    n <- length(data)
    sample_mean <- mean(data)
    sample_variance <- var(data)
    shape_mom <- 0.5*(1-sample_mean**2/sample_variance)
    scale_mom <- 0.5*sample_mean*(1+sample_mean**2/sample_variance)
    
    if (shape_mom < 0.25) {
        # [shape, scale]
        Sigma_mom <- matrix(
            1/n*(1-shape_mom)**2/(1-2*shape_mom)/(1-3*shape_mom)/(1-4*shape_mom)*c((1-2*shape_mom)**2*(1-shape_mom+6*shape_mom**2), 
                                                                                    -scale_mom*(1-2*shape_mom)*(1-4*shape_mom+12*shape_mom**2),
                                                                                    -scale_mom*(1-2*shape_mom)*(1-4*shape_mom+12*shape_mom**2),
                                                                                    2*scale_mom**2*(1-6*shape_mom+12*shape_mom**2)),
            nrow=2, ncol=2
        )
    } else {
        Sigma_mom <- NA
    }

    # p_vector = (seq(1,n) - 0.35)/n
    # t = sum((1-p_vector)*sort(data))/n
    t = sum((n - seq(1, n))/(n-1)*sort(data))/n
    shape_pwm <- 2 - sample_mean/(sample_mean-2*t)
    scale_pwm <- 2*sample_mean*t/(sample_mean-2*t)

    if (shape_pwm < 0.5) {
        # [shape, scale]
        Sigma_pwm <- matrix(
            1/n/(1-2*shape_pwm)/(3-2*shape_pwm)*c((1-shape_pwm)*(2-shape_pwm)**2*(1-shape_pwm+2*shape_pwm**2), 
                                                   -scale_pwm*(2-shape_pwm)*(2-6*shape_pwm+7*shape_pwm**2-2*shape_pwm**3),
                                                   -scale_pwm*(2-shape_pwm)*(2-6*shape_pwm+7*shape_pwm**2-2*shape_pwm**3),
                                                   scale_pwm**2*(7-18*shape_pwm+11*shape_pwm**2-2*shape_pwm**3)),
            nrow=2, ncol=2
        )
    } else {
        Sigma_pwm <- NA
    }
    fitGPD <- list(shape_mom=shape_mom, scale_mom=scale_mom, Sigma_mom=Sigma_mom, 
                   shape_pwm=shape_pwm, scale_pwm=scale_pwm, Sigma_pwm=Sigma_pwm,
                   prob_over_threshold=prob_over_threshold, nexc=n)
    h <- function(shape, scale) {
        if (rhs == Inf) {
            return(1-POT::pgpd(q=lhs-u,scale=scale, shape=shape))
        }
        return(POT::pgpd(q=rhs-u, scale=scale, shape=shape) - POT::pgpd(q=lhs-u,scale=scale, shape=shape))
    }
    hGrad <- function(shape, scale) {
        if (rhs == Inf) {
            return(-gradientpGPD(x=lhs-u, xi=shape, beta=scale))
        }
        return(gradientpGPD(x=rhs-u, xi=shape, beta=scale) - gradientpGPD(x=lhs-u, xi=shape, beta=scale))
    }
	
	Ubd <- asymptoticCIforGPD_delta(fitGPD, h, hGrad, alpha = 1 - conf, verbose = FALSE)

	return(list(CI=Ubd))
}

test <- function() {
    success_cnt <- 0
    scale_true <- 1
    shape_true <- 0.1
    nsample <- 50000
    num_rep <- 100
    lhs <- POT::qgpd(p=0.99, loc=0, scale=scale_true, shape=shape_true)
    rhs <- POT::qgpd(p=0.995, loc=0, scale=scale_true, shape=shape_true)
    tail_prob_true <- POT::pgpd(q=rhs, loc=0, scale=scale_true, shape=shape_true) -
                    POT::pgpd(q=lhs, loc=0, scale=scale_true, shape=shape_true)
    conf <- 0.95
    set.seed(20)
    for (index in 1:num_rep) {
    data <- POT::rgpd(nsample, loc = 0, scale = scale_true, shape = shape_true) 
    out <- gpdTIP(data, lhs, rhs, conf)
    writeLines(sprintf("True value is %f", tail_prob_true))
    writeLines(sprintf("95%% confidence interval is [%f, %f]", out$lB, out$uB))
    if (out$lB < tail_prob_true && out$uB > tail_prob_true) {
        success_cnt <- success_cnt + 1 
        }  
    }
    writeLines(sprintf("coverage probability: %f", success_cnt/num_rep))
}
