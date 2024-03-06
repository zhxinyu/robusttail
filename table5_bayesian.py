## Use Bayesian method to estimate CI.
import dataPreparationUtils as dpu
from scipy.stats import gamma, lognorm, pareto
import os
try:
	from rpy2.robjects.packages import importr
except:
    os.environ['R_HOME'] = os.path.join(os.environ['CONDA_PREFIX'], 'lib', 'R')
    from rpy2.robjects.packages import importr
    
import rpy2.robjects as ro
import pandas as pd
import numpy as np
from rpy2.robjects import numpy2ri
numpy2ri.activate()

importr('base')
utils = importr('utils')
try:
	importr('POT')
	importr('nloptr')    
	importr('MASS')
	importr('eva')
except:
	utils.install_packages('POT', contribulr="https://cran.microsoft.com/")
	utils.install_packages('nloptr', contribulr="https://cran.microsoft.com/")
	utils.install_packages('MASS', contribulr="https://cran.microsoft.com/")
	utils.install_packages('eva', contribulr="https://cran.microsoft.com/")
	importr('POT')
	importr('nloptr')    
	importr('MASS')    
	importr('eva')
 

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

eval_tip <- function(x, data, u, lhs, rhs) {
  scale <- x[1]
  shape <- x[2]
  tail_internval_probability <- pgpd(rhs, u, scale, shape) - pgpd(lhs, u, scale, shape)
  return(tail_internval_probability)
}

sampleProposal <- function(x0, prior_scale, prior_shape, u) {
  # Generate samples
  repeat {
    xstar <- mvrnorm(1, mu = x0, Sigma = matrix(c(1, 0, 0, 1), nrow = 2))
    if (xstar[1] > prior_scale[1] && xstar[1] < prior_scale[2] &&
        xstar[2] > prior_shape[1] && xstar[2] < prior_shape[2]
    ) {
      # notice the proposal probability distributio ratio is equal to 1 so we skip the term
      # in the code, i.e. Q(xstar|x0) = Q(x0|xstar) if Q(x'|x) ~ N(x, I). 
      accrej <- exp( min(0, -gpd_neg_loglikelihood(xstar, data, u) + 
                             gpd_neg_loglikelihood(x0,    data, u))
      )
      if (runif(1, min = 0, max = 1) < accrej) {
        return(xstar)
      }
      
    }
  }
}

estimateTIPviaMCMC <- function(x0, prior_scale, prior_shape, 
                               data, u, lhs, rhs, conf) {
  burnIn <- 20000
  for ( i in 1:burnIn ){
    x0 <- sampleProposal(x0, prior_scale, prior_shape, u)
  }
  chainLength <- 1000
  tip <- rep(NA, chainLength)
  tip_idx <- 1 
  base_probability <- sum(data>u)/length(data)   
  for ( i in 1 : chainLength  ){
    tip[tip_idx] = base_probability*eval_tip(x=x0, data=data, u=u, lhs=lhs, rhs=rhs)
    x0 <- sampleProposal(x0, prior_scale, prior_shape, u)
    tip_idx = tip_idx + 1    
  }
  return(quantile(tip, probs = c(0, conf), na.rm=TRUE))
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
  # Find initial point for the following MCMC method to estimate credible interval for TIP.
  suppressWarnings(z <- POT::fitgpd(data, threshold = u, est = 'mle'))
  x0 = c(as.numeric(z$param[1]), as.numeric(z$param[2]))
  estimate <- NaN

  prior_scale = c(max(0, x0[1] -1), x0[1] + 1)
  prior_shape = c(x0[2] - 1, x0[2] + 1)

  bound_val <- estimateTIPviaMCMC(x=x0, prior_scale=prior_scale, prior_shape=prior_shape,
                                  data=data, u=u, lhs=lhs, rhs=rhs, conf=.95)

  CI <- c(bound_val[1], bound_val[2])
  out <- list(estimate, CI, conf)
  names(out) <- c("Estimate", "CI", "ConfLevel")
  out
}

    '''

if __name__ == '__main__':
    import argparse  
    FILE_DIR = "large_bayesian"
    
    parser = argparse.ArgumentParser(description='TIP estimation using MCMC.')
    parser.add_argument('lhs_st', type=float, help='start of LHS list in the objective function')
    parser.add_argument('ds', type=str, help='Data source for simulation')
    args = parser.parse_args()
    
    randomSeed = 20220222
    stringToDataModule = {"gamma": gamma,
						  "lognorm": lognorm,
						  "pareto": pareto}
    trueValue = 0.005
    # percentageLHSs = np.linspace(0.9, 0.99, 10).tolist()
    # dataSources = ['gamma','lognorm','pareto']
    percentageLHSs = [args.lhs_st]
    # percentageLHSs = np.arange(args.lhs_st, args.lhs_ed, 0.01)
    dataSources = [ args.ds ]
    metaDataDict = {"dataSize": 500}
    
    nrep = 200
    result = []
    columns = ["Data Source", 'nData',"percentageLHS", "Lower Bound","Upper Bound", "True Value", "Repetition Index"]
    for percentageLHS in percentageLHSs:
        percentageLHS = np.round(percentageLHS,2)
        percentageRHS = percentageLHS + trueValue
        for dataSource in dataSources:
            dataModule = stringToDataModule[dataSource]
            leftEndPointObjective = dpu.endPointGeneration(
                dataModule, percentageLHS, dpu.dataModuleToDefaultParamDict[dataModule])
            rightEndPointObjective = dpu.endPointGeneration(
                dataModule, percentageRHS, dpu.dataModuleToDefaultParamDict[dataModule])
            for nnrep in range(nrep):
                print(f"Working on {percentageLHS}_{dataSource}_{nnrep}")
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
if ("error" %in% class(out)) {{
  print(out)
}}

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
            df.to_csv(os.path.join(FILE_DIR, f'table5_{percentageLHS}_{dataSource}.csv'), header=columns, index = False)
    df = pd.DataFrame(data = result, columns = columns)

    # df.to_csv(os.path.join(FILE_DIR, f'tableFive_bayesian_{args.lhs_st}_{args.ds}.csv'), header=columns, index = False)
