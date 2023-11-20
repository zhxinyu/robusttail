#' GPD Tail Probability and Confidence Interval for Stationary Models
#'
#' Computes stationary tail probability and confidence interval for the Generalized Pareto distribution,
#' using either the delta method or profile likelihood.
#'
#' @param z An object of class `gpdFit'.
#' @param lhs Left end point `lhs` for the tail probability to be estimated, i.e. P(X>=lhs).
#' @param conf Confidence level. Defaults to 95 percent.
#' @param method The method to compute the confidence interval - either delta method (default) or profile likelihood.
#' @param plot Plot the profile likelihood and estimate (vertical line)?
#' @param opt Optimization method to maximize the profile likelihood if that is selected. Argument passed to optim. The
#' default method is Nelder-Mead.
#'
#' @references Coles, S. (2001). An introduction to statistical modeling of extreme values (Vol. 208). London: Springer.
#' @examples
#' x <- rgpd(5000, loc = 0, scale = 1, shape = -0.1)
#' ## Compute 50-period return level.
#' z <- gpdFit(x, nextremes = 200)
#' gpdTP(z, 2, 3, method = "delta")
#' gpdTP(z, 2, 3, method = "profile")
#' @return
#' \item{Estimate}{Estimated tail interval probability.}
#' \item{CI}{Confidence interval for the tail interval probability.}
#' \item{ConfLevel}{The confidence level used.}
#' @details Caution: The profile likelihood optimization may be slow for large datasets.
#' @export
#' 
#' 

rm(list=ls()) 
gpdTP <- function(z, lhs, conf = .95, method = c("delta", "profile"), plot = TRUE) {
  
  method <- match.arg(method)
  u <- z$threshold
  scale_0 <- as.numeric(z$param[1]) # scale
  shape_0 <- as.numeric(z$param[2]) # shape 
  rel_tp_0   <- as.numeric(1 - POT::pgpd(q=lhs, loc = u, scale = scale_0, shape = shape_0))

  if(method == "delta") {
    stop("Delta method has not been implemented yet.")
  } else {
    gpdLik <- function(shape, xtp) {
      # sprintf("gpdLik:: %f", shape)
      if(shape == 0) {
        scale <- (lhs - u)/log(1/xtp)
      } else {
        scale <- (lhs - u)*shape/(xtp**(-shape)-1)
      }

      if(scale <= 0) {
        out <- .Machine$double.xmax
      } else {
        out <- POT::dgpd(z$data[z$data > u], loc = u, scale = scale, shape = shape, log = TRUE)
        ## multiply by -1 so as to pass in the `optim` function for minimization
        out <- - sum(out)
        if(out == Inf)
          out <- .Machine$double.xmax
      }
      out
    }
    cutoff <- qchisq(conf, 1)
    lmax <- sum(POT::dgpd(z$data[z$data > u], loc = u, scale = scale_0, shape = shape_0, log = TRUE))
    stats_diff <- function(xtp) {
      # sprintf("prof:: %f", xtp)
      multiplier = 10
      shrink_ratio = 0.9
      min_value = .Machine$double.xmax
      minima = 0
      # find the minimum value for sup_{\theta\in \Theta_0} \mathcal{L}(\theta)
      for (index in 1:100) {
        yes <- optimize(f=gpdLik, interval=c(0, shape_0*multiplier), xtp = xtp)
        min_value = yes$objective
        if (min_value<.Machine$double.xmax){
          minima = yes$minimum
          break
        }
        multiplier = multiplier*shrink_ratio        
      }
      if (min_value == .Machine$double.xmax) {
        writeLines(sprintf("Unable to solve the optimization problem"))
        return(NA)
      }
      # sprintf("prof::end")
      best_shape <- minima
      lci <- -min_value
      2*(lmax-lci) - cutoff
    }
    stats_diff <- Vectorize(stats_diff)
    out1 <- NULL
    out2 <- NULL
    try(out1 <- uniroot(stats_diff, c(1e-6, rel_tp_0), tol=1e-6))
    if (is.null(out1)) {
      out1$root = rel_tp_0
    }
    try(out2 <- uniroot(stats_diff, c(rel_tp_0, 1- 1e-6), tol=1e-6))
    if (is.null(out2)) {
      out2$root = rel_tp_0
    }
    # suppressWarnings()
    # suppressWarnings()
    CI <- c(min(out1$root, out2$root), max(out1$root, out2$root))
    # stats_diff(tp_0)

    if(plot) {
      stats_diff1 <- function(xtp) {- stats_diff(xtp)}
      # curve(stats_diff1, from=tp_0/2, to=tp_0*2, xlab = 'Tail Interval Probability', ylab = 'LRT - Cutoff')
      suppressWarnings(curve(stats_diff1, from = CI[1], to = CI[2], n = 50, xlab = 'Tail Probability', ylab = 'LRT - Cutoff'))
      abline(v = rel_tp_0, col = "blue")
    }
  }
  out <- list(z$pat*rel_tp_0, z$pat*CI, conf)
  names(out) <- c("Estimate", "CI", "ConfLevel")
  out
}

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

success_cnt <- 0
scale_true <- 1
shape_true <- 1.5
set.seed(42)
lhs <- POT::qgpd(p=0.995, loc = 0, scale = scale_true, shape = shape_true)
tail_prob_true <- 1 - POT::pgpd(q=lhs, loc = 0, scale = scale_true, shape = shape_true)

num_rep = 1
for (index in 1:num_rep) {
  data <- POT::rgpd(500, loc = 0, scale = scale_true, shape = shape_true) 
  u <- gof_find_threshold(data)
  if (is.null(u)) {
    u <- gof_find_threshold(data, bootstrap=TRUE)
  }

  z <- POT::fitgpd(data, threshold = u, est = 'mle')
  # pat <- z$pat
  scale_0 <- as.numeric(z$param[1]) # scale
  shape_0 <- as.numeric(z$param[2]) # shape 
  # tail_prob_0   <- pat*as.numeric(1 - POT::pgpd(q=lhs, loc = u, scale = scale_0, shape = shape_0))
  method <- "profile"
  conf <- .95
  out <- gpdTP(z, lhs, method=method, conf=conf, plot=FALSE)
  writeLines(sprintf("True value is %f", tail_prob_true))
  writeLines(sprintf("95%% confidence interval is [%f, %f]", out$CI[1], out$CI[2]))
  if (out$CI[1] < tail_prob_true && out$CI[2] > tail_prob_true) {
    success_cnt <- success_cnt + 1 
  }
}
writeLines(sprintf("coverage probability: %f", success_cnt/num_rep))

POT::gpd.pfshape(z, c(0, 10), conf = 0.95)
0.4545455 2.3737374