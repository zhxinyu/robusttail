from optimization_unit import Optimization_Plain_ChiSquare, Optimization_Monetone_ChiSquare, Optimization_Convex_ChiSquare, Optimization_Plain_Kolmogorov, Optimization_Monotone_Kolmogorov, Optimization_Convex_Kolmogorov
from optimization_engine import PolynomialFunction
import numpy as np
import typing


def OptimizationWithRectangularConstraint(D: int, inputData: np.ndarray,
                                          thresholdPercentage: typing.Union[float, typing.List[float]],
                                          alpha: float,
                                          leftEndPointObjective: float, rightEndPointObjective: float,
                                          bootstrappingSize: int, bootstrappingSeed: int):
    #############################################################################################################
    #############################################################################################################
    # Rectangular constraint
    # confidence level = 1-alpha
    #############################################################################################################
    #############################################################################################################

    threshold = np.quantile(inputData, thresholdPercentage)
    if type(thresholdPercentage) == float:
        numMultiThreshold = 1
    else:
        numMultiThreshold = len(thresholdPercentage)
    h = PolynomialFunction(
        [leftEndPointObjective, rightEndPointObjective, np.inf], [[1], [0]])
    if D == 0:
        return Optimization_Plain_Kolmogorov(data=inputData,
                                             threshold=threshold,
                                             ObjectiveFunction=h,
                                             bootstrappingSize=bootstrappingSize,
                                             bootstrappingSeed=bootstrappingSeed,
                                             alpha=alpha,
                                             numMultiThreshold=numMultiThreshold)
    if D == 1:
        return Optimization_Monotone_Kolmogorov(data=inputData,
                                                threshold=threshold,
                                                ObjectiveFunction=h,
                                                bootstrappingSize=bootstrappingSize,
                                                bootstrappingSeed=bootstrappingSeed,
                                                alpha=alpha,
                                                numMultiThreshold=numMultiThreshold)
    if D == 2:
        return Optimization_Convex_Kolmogorov(data=inputData,
                                              threshold=threshold,
                                              ObjectiveFunction=h,
                                              bootstrappingSize=bootstrappingSize,
                                              bootstrappingSeed=bootstrappingSeed,
                                              alpha=alpha,
                                              numMultiThreshold=numMultiThreshold)


def OptimizationWithEllipsodialConstraint(D: int, inputData: np.ndarray,
                                          thresholdPercentage: typing.Union[float, typing.List[float]],
                                          alpha: float,
                                          leftEndPointObjective: float, rightEndPointObjective: float,
                                          gEllipsoidalDimension: int,
                                          bootstrappingSize: int, bootstrappingSeed: int):
    #############################################################################################################
    #############################################################################################################
    # Ellipsodial constraint
    # confidence level = 1-alpha
    #############################################################################################################
    #############################################################################################################
    threshold = np.quantile(inputData, thresholdPercentage)
    h = PolynomialFunction(
        [leftEndPointObjective, rightEndPointObjective, np.inf], [[1], [0]])
    g_Es = [PolynomialFunction([threshold, np.inf], [[0] * i + [1]])
            for i in range(gEllipsoidalDimension)]

    if type(thresholdPercentage) == float:
        numMultiThreshold = 1
    else:
        numMultiThreshold = len(thresholdPercentage)

    mu = np.array([np.sum(inputData**power*(inputData > threshold)) /
                  inputData.size for power in range(gEllipsoidalDimension)])
    Sigma = np.cov(np.vstack(
        [(inputData > threshold)*1.0*inputData**power for power in range(gEllipsoidalDimension)]))

    if D == 0:
        return Optimization_Plain_ChiSquare(data=inputData,
                                            threshold=threshold,
                                            ObjectiveFunction=h,
                                            MomentConstraintFunctions=g_Es,
                                            mu=mu,
                                            Sigma=Sigma,
                                            bootstrappingSize=bootstrappingSize,
                                            bootstrappingSeed=bootstrappingSeed,
                                            alpha=alpha,
                                            numMultiThreshold=numMultiThreshold)
    if D == 1:
        return Optimization_Monetone_ChiSquare(data=inputData,
                                               threshold=threshold,
                                               ObjectiveFunction=h,
                                               MomentConstraintFunctions=g_Es,
                                               mu=mu,
                                               Sigma=Sigma,
                                               bootstrappingSize=bootstrappingSize,
                                               bootstrappingSeed=bootstrappingSeed,
                                               alpha=alpha,
                                               numMultiThreshold=numMultiThreshold)
    if D == 2:
        return Optimization_Convex_ChiSquare(data=inputData,
                                             threshold=threshold,
                                             ObjectiveFunction=h,
                                             MomentConstraintFunctions=g_Es,
                                             mu=mu,
                                             Sigma=Sigma,
                                             bootstrappingSize=bootstrappingSize,
                                             bootstrappingSeed=bootstrappingSeed,
                                             alpha=alpha,
                                             numMultiThreshold=numMultiThreshold)


if __name__ == '__main__':
    pass
