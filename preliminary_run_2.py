from optimization_unit import Optimization_Plain_ChiSquare, Optimization_Monetone_ChiSquare, Optimization_Convex_ChiSquare, Optimization_Plain_Kolmogorov, Optimization_Monotone_Kolmogorov, Optimization_Convex_Kolmogorov
from optimization_engine import PolynomialFunction
from scipy.stats import norm
import numpy as np



if __name__ == '__main__':
    #############################################################################################################
    #############################################################################################################
    #### Ellipsodial constraint
    #############################################################################################################
    #############################################################################################################
    sizeOnData = 500
    data = norm.rvs(size = sizeOnData,random_state=42)
    quantile_threshold = 0.7
    threshold = np.quantile(data,quantile_threshold)
    alpha = 0.05
    L = 2
    R = 3
    gEllipsoidalDimension = 4
    bootstrappingSize = 500
    numMultiThreshold = 1
    h = PolynomialFunction([L, R, np.inf], [[1], [0]])
    g_Es = [PolynomialFunction([threshold, np.inf], [[0] * i + [1]]) for i in range(gEllipsoidalDimension)]
    mu = np.array([np.sum(data**power*(data>threshold))/data.size for power in range(gEllipsoidalDimension)])
    Sigma = np.cov(np.vstack([(data>threshold)*1.0*data**power for power in range(gEllipsoidalDimension)]))
    bootstrappingSeed = 20220413
    output = []
    output.append(Optimization_Plain_ChiSquare(data = data,
                                 threshold = threshold,
                                 ObjectiveFunction = h,
                                 MomentConstraintFunctions = g_Es,
                                 mu = mu,
                                 Sigma = Sigma,
                                 bootstrappingSize = bootstrappingSize,
                                 bootstrappingSeed = bootstrappingSeed,
                                 alpha = alpha,
                                 numMultiThreshold = numMultiThreshold))
    bootstrappingSeed+=1
    output.append(Optimization_Monetone_ChiSquare(data = data,
                                 threshold = threshold,
                                 ObjectiveFunction = h,
                                 MomentConstraintFunctions = g_Es,
                                 mu = mu,
                                 Sigma = Sigma,
                                 bootstrappingSize = bootstrappingSize,
                                 bootstrappingSeed = bootstrappingSeed,
                                 alpha = alpha,
                                 numMultiThreshold = numMultiThreshold))
    bootstrappingSeed+=1
    output.append(Optimization_Convex_ChiSquare(data = data,
                                 threshold = threshold,
                                 ObjectiveFunction = h,
                                 MomentConstraintFunctions = g_Es,
                                 mu = mu,
                                 Sigma = Sigma,
                                 bootstrappingSize = bootstrappingSize,
                                 bootstrappingSeed = bootstrappingSeed,
                                 alpha = alpha,
                                 numMultiThreshold = numMultiThreshold))
    print(output)
    #############################################################################################################
    #############################################################################################################
    #### Rectangular constraint
    #############################################################################################################
    #############################################################################################################

    sizeOnData = 500
    data = norm.rvs(size = sizeOnData,random_state=42)
    quantile_threshold = 0.7
    threshold = np.quantile(data,quantile_threshold)
    alpha = 0.05
    L = 2
    R = 3
    bootstrappingSize = 500
    numMultiThreshold = 1
    h = PolynomialFunction([L, R, np.inf], [[1], [0]])
    output = []
    bootstrappingSeed = 20220413
    output.append(Optimization_Plain_Kolmogorov(data = data,
                                  threshold = threshold,
                                  ObjectiveFunction = h,
                                  bootstrappingSize = bootstrappingSize,
                                  bootstrappingSeed = bootstrappingSeed,
                                  alpha = alpha,
                                  numMultiThreshold = numMultiThreshold))

    bootstrappingSeed += 1
    output.append(Optimization_Monotone_Kolmogorov(data = data,
                                  threshold = threshold,
                                  ObjectiveFunction = h,
                                  bootstrappingSize = bootstrappingSize,
                                  bootstrappingSeed = bootstrappingSeed,
                                  alpha = alpha,
                                  numMultiThreshold = numMultiThreshold))

    bootstrappingSeed += 1
    output.append(Optimization_Convex_Kolmogorov(data = data,
                                  threshold = threshold,
                                  ObjectiveFunction = h,
                                  bootstrappingSize = bootstrappingSize,
                                  bootstrappingSeed = bootstrappingSeed,
                                  alpha = alpha,
                                  numMultiThreshold = numMultiThreshold))
    print(output)
