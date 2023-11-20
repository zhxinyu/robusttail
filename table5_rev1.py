## Use profile likelihood to estimate CI.
import dataPreparationUtils as dpu
from scipy.stats import gamma, lognorm, pareto
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import pandas as pd
import numpy as np
import os
from rpy2.robjects import numpy2ri
numpy2ri.activate()

importr('base')
utils = importr('utils')
try:
	importr('POT')
	importr('nloptr')    
except:
	utils.install_packages('POT', contribulr="https://cran.microsoft.com/")
	utils.install_packages('nloptr', contribulr="https://cran.microsoft.com/")
	importr('POT')
	importr('nloptr')    

POTUtilityinR='''
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
gpd_neg_loglikelihood <- function(x, data, u) {
  scale <- x[1]
  shape <- x[2]
  xx_u <- data[data > u] - u
  if (shape == 0) {
    out <- (-log(scale)-xx_u/scale)
  } else {
    out <- (-log(scale)-(1+1/shape)*log(1+shape/scale*xx_u))
  }
  out <- -sum(out)
  if (is.na(out)){
    out <-.Machine$double.xmax
  }
  return(out)
}
gpd_neg_loglikelihood_derivative <- function(x, data, u) {
  scale <- x[1]
  shape <- x[2]  
  xx_u <- data[data > u] - u
  dlll_dscale <- -1/scale +(1+shape)*xx_u/(scale**2 + scale*shape*xx_u)
  dlll_dshape <- 1/shape**2*log(1+shape*xx_u/scale) -(1+shape)*xx_u/(shape*scale+shape**2*xx_u)
  out_dscale <- -sum(dlll_dscale)
  out_dshape <- -sum(dlll_dshape)
  out <- c(out_dscale, out_dshape)
  return(out)
}
pgpd <- function(x, loc, scale, shape) {
  if (shape == 0) {
    return(1-log(-(x-loc)/scale))
  } else {
    return(1-(1+shape/scale*(x-loc))**(-1/shape))
  }
}
dgpd <-function(x, loc, scale, shape) {
  if (shape == 0) {
    return(1/scale*exp(-(x-loc)/scale))
  } else {
    return(1/scale*(1+shape/scale*(x-loc))**(-1/shape-1))
  }  
}
d_pgpd_d_scale <- function(x, loc, scale, shape) {
  return(-dgpd(x, loc, scale, shape)*(x-loc)/scale)
}
d_pgpd_d_shape <- function(x, loc, scale, shape) {
  first_part <- -(1-pgpd(x, loc, scale, shape))/shape**2*log(1+shape*(x-loc)/scale)
  second_part <- dgpd(x, loc, scale, shape)*(x-loc)/shape
  return(first_part+second_part)
}
eval_llh_f <- function(x, data, u) {
  objective <- gpd_neg_loglikelihood(x, data, u)
  grad <- gpd_neg_loglikelihood_derivative(x, data, u)
  return( list( "objective"=objective, "gradient"=grad ) )  
}
eval_llh_g_ineq <- function(x, data, u) {
    xx_u = data[data>u] -u
    scale <- x[1]
    shape <- x[2]
    constr1 <- -shape*(xx_u)-scale
    grad1_scale  <- -rep(1, length(xx_u))
    grad1_shape  <- -xx_u
    constr <- c(constr1)
    grad   <- cbind(grad1_scale, grad1_shape)
    return( list( "constraints"=constr, "jacobian"=grad ) )
}
eval_tip_f <- function(x, data, u, direction, lhs, rhs, llmax, cutoff) {
  scale <- x[1]
  shape <- x[2]
  tail_internval_probability <- pgpd(rhs, u, scale, shape) - pgpd(lhs, u, scale, shape)
  objective <- direction*tail_internval_probability
  dout_dscale <- direction*(d_pgpd_d_scale(rhs, u, scale, shape) - d_pgpd_d_scale(lhs, u, scale, shape))
  dout_dshape <- direction*(d_pgpd_d_shape(rhs, u, scale, shape) - d_pgpd_d_shape(lhs, u, scale, shape))
  grad <- c(dout_dscale, dout_dshape)
  return( list( "objective"=objective, "gradient"=grad ) )
}
eval_tip_g_ineq <- function(x, data, u, direction, lhs, rhs, llmax, cutoff) {
    xx_u = data[data>u] -u
    scale <- x[1]
    shape <- x[2]
    constr1 <- (-2) *( -gpd_neg_loglikelihood(x=x, data=data, u=u) - llmax ) - cutoff
    constr2 <- -shape*(xx_u)-scale
    grad1   <- 2*gpd_neg_loglikelihood_derivative(x=x, data=data, u=u)
    grad2_scale  <- -rep(1, length(xx_u))
    grad2_shape  <- -xx_u
    constr <- c(constr1, constr2)
    grad   <- rbind(grad1, cbind(grad2_scale, grad2_shape))
    return( list( "constraints"=constr, "jacobian"=grad ) )
}
gpdTIP <- function(data, lhs, rhs, conf = .95) {

  if(lhs >= rhs) {
    stop("Left end point must be smaller than right end point!")
  }

  u <- gof_find_threshold(data)
  if (is.null(u)) {
    u <- gof_find_threshold(data, bootstrap=TRUE)
  }
  base_probability <- sum(data>u)/length(data) 
  # Find initial point for the following non-linear optimization problem
  suppressWarnings(z <- POT::fitgpd(data, threshold = u, est = 'mle'))
  x0 = c(as.numeric(z$param[1]), as.numeric(z$param[2]))

  ## Find the maximum likelihood estimation and log likelihood value. 
  ## use either NLOPT_LD_SLSQP, LD_MMA
  local_opts1 <- list(   "algorithm"  = "NLOPT_LD_MMA",
                        "xtol_rel"   = 0,
                        "xtol_abs"   = 1e-8,
                        "maxeval"    = 0,
                        "check_derivatives" = FALSE,
                        "check_derivatives_print" = "errors",                        
                        "check_derivatives_tol" = 1e-04)
  local_res <- nloptr::nloptr(x0=x0, 
                              eval_f=eval_llh_f, 
                              lb=c(0, -Inf), ub=c(Inf, Inf), 
                              eval_g_ineq=eval_llh_g_ineq,
                              opts=local_opts1,
                              data=data, u=u)
  llmax <- -local_res$objective
  x0 <- c(local_res$solution[1], local_res$solution[2])  
  cutoff <- qchisq(p=conf, df=1)
  estimate <- base_probability*eval_llh_f(x=x0, data=data, u=u)$objective

  ## NLOPT_LD_SLSQP, LD_MMA
  local_opts2 <- list(   "algorithm"  = "NLOPT_LD_MMA",
                        "xtol_rel"   = 0,
                        "xtol_abs"   = 1e-8,
                        "maxeval"    = 0,
                        "check_derivatives" = FALSE,
                        "check_derivatives_print" = "errors",
                        "check_derivatives_tol" = 1e-4)
  direction <- 1
  opt_res <- nloptr::nloptr(x0=x0,
                        eval_f=eval_tip_f,
                        lb=c(0, -Inf),
                        ub=c(Inf, Inf),
                        eval_g_ineq=eval_tip_g_ineq,
                        opts=local_opts2,
                        data=data, u=u, direction=direction, lhs=lhs, rhs=rhs, llmax=llmax, cutoff=cutoff)
  local_min_val <- direction*opt_res$objective*base_probability

  direction <- -1
  opt_res <- nloptr::nloptr(x0=x0,
                        eval_f=eval_tip_f,
                        lb=c(0, -Inf),
                        ub=c(Inf, Inf),
                        eval_g_ineq=eval_tip_g_ineq,
                        opts=local_opts2,
                        data=data, u=u, direction=direction, lhs=lhs, rhs=rhs, llmax=llmax, cutoff=cutoff)
  local_max_val <- direction*opt_res$objective*base_probability  
   
  CI <- c(local_min_val, local_max_val)
  out <- list(estimate, CI, conf)
  names(out) <- c("Estimate", "CI", "ConfLevel")
  return(out)
}

    '''

if __name__ == '__main__':
	randomSeed = 20220222
	stringToDataModule = {"gamma": gamma,
						"lognorm": lognorm,
						"pareto": pareto}

	trueValue = 0.005
	percentageLHSs = np.linspace(0.9, 0.99, 10).tolist()

	metaDataDict = {"dataSize": 500}
	dataSources = ['gamma','lognorm','pareto']
	nrep = 200
	result = []
	columns = ["Data Source", 'nData',"percentageLHS", "Lower Bound","Upper Bound", "True Value", "Repetition Index"]
	for percentageLHS in percentageLHSs:
		percentageRHS = percentageLHS + trueValue
		for dataSource in dataSources:
			dataModule = stringToDataModule[dataSource]
			leftEndPointObjective = dpu.endPointGeneration(
				dataModule, percentageLHS, dpu.dataModuleToDefaultParamDict[dataModule])
			rightEndPointObjective = dpu.endPointGeneration(
				dataModule, percentageRHS, dpu.dataModuleToDefaultParamDict[dataModule])
			for nnrep in range(nrep):
				try:
					metaDataDict['random_state'] = randomSeed+nnrep
					inputData = dpu.RawDataGeneration(dataModule, 
													dpu.dataModuleToDefaultParamDict[dataModule], 
													metaDataDict['dataSize'], 
													metaDataDict['random_state'])
					POTApply = '''
					lhs <- {:}

					rhs <- {:}

					data <- c({:})

					out <- tryCatch(
					gpdTIP(data, lhs, rhs, conf=0.95), 
					error = function(e) e
					)

					bbd <- if ("error" %in% class(out)) NA else{{
						out$CI
					}}
					bbd 
					'''.format(leftEndPointObjective, 
							rightEndPointObjective, 
							', '.join([str(eachData)for eachData in inputData.tolist()]))
					roResult = ro.r(POTUtilityinR+POTApply)
					result.append([dataSource, metaDataDict['dataSize'], percentageLHS, roResult[0], roResult[1], trueValue, nnrep])
					print(result[-1])
				except Exception as e: 
					print(e)
					result.append([dataSource, metaDataDict['dataSize'], 
								percentageLHS, 0, 
								0, 
								trueValue, nnrep])
			
			print(f"Finish experiments on {percentageLHS}-{dataSource}")
   
	df = pd.DataFrame(data = result, columns = columns)

	FILE_DIR = "large"
	df.to_csv(os.path.join(FILE_DIR, 'tableFive_rev1.csv'), header=columns, index = False)
