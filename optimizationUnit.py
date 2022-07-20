from optimizationEngine import optimization, PolynomialFunction
from calibration import etaSpecification, nuSpecification, zOfChiSquare, zOfKolmogorov
import numpy as np
from copy import deepcopy
import typing


def OptimizationPlainChiSquare(data: np.ndarray,
                               threshold: float,
                               ObjectiveFunction: PolynomialFunction,
                               MomentConstraintFunctions: typing.List[PolynomialFunction],
                               mu: np.ndarray,
                               Sigma: np.ndarray,
                               bootstrappingSize: int,
                               bootstrappingSeed: int,
                               alpha: float,
                               numMultiThreshold: int) -> float:
    ## (0, chi2)
    D_riser_number = 0
    gEllipsoidalDimension = len(MomentConstraintFunctions)
    z = zOfChiSquare(alpha=alpha,
                     D_riser_number=D_riser_number,
                     gDimension=gEllipsoidalDimension,
                     numMultiThreshold=numMultiThreshold)
    radius = z/data.shape[0]
    return optimization(D_riser_number=D_riser_number,
                        threshold_level=threshold,
                        h=ObjectiveFunction,
                        g_Es=MomentConstraintFunctions,
                        mu_value=mu, Sigma=Sigma, radius=radius)


def OptimizationMonetoneChiSquare(data: np.ndarray,
                                  threshold: float,
                                  ObjectiveFunction: PolynomialFunction,
                                  MomentConstraintFunctions: typing.List[PolynomialFunction],
                                  mu: np.ndarray,
                                  Sigma: np.ndarray,
                                  bootstrappingSize: int,
                                  bootstrappingSeed: int,
                                  alpha: float,
                                  numMultiThreshold: int) -> float:
    ## (1, chi2)
    D_riser_number = 1
    gEllipsoidalDimension = len(MomentConstraintFunctions)
    eta = etaSpecification(data=data,
                           threshold=threshold,
                           alpha=alpha,
                           bootstrappingSize=bootstrappingSize,
                           bootstrappingSeed=bootstrappingSeed,
                           D_riser_number=D_riser_number,
                           numMultiThreshold=numMultiThreshold)
    z = zOfChiSquare(alpha=alpha,
                     D_riser_number=D_riser_number,
                     gDimension=gEllipsoidalDimension,
                     numMultiThreshold=numMultiThreshold)
    radius = z/data.shape[0]
    return optimization(D_riser_number=D_riser_number,
                        eta=eta,
                        threshold_level=threshold,
                        h=ObjectiveFunction,
                        g_Es=MomentConstraintFunctions,
                        mu_value=mu, Sigma=Sigma, radius=radius)


def OptimizationConvexChiSquare(data: np.ndarray,
                                threshold: float,
                                ObjectiveFunction: PolynomialFunction,
                                MomentConstraintFunctions: typing.List[PolynomialFunction],
                                mu: np.ndarray,
                                Sigma: np.ndarray,
                                bootstrappingSize: int,
                                bootstrappingSeed: int,
                                alpha: float,
                                numMultiThreshold: int) -> float:
    ## (2, chi2)
    D_riser_number = 2
    gEllipsoidalDimension = len(MomentConstraintFunctions)

    [eta_lb, eta_ub] = etaSpecification(data=data,
                                        alpha=alpha,
                                        threshold=threshold,
                                        bootstrappingSize=bootstrappingSize,
                                        bootstrappingSeed=bootstrappingSeed,
                                        D_riser_number=D_riser_number,
                                        numMultiThreshold=numMultiThreshold)

    nu = nuSpecification(data=data,
                         threshold=threshold,
                         alpha=alpha,
                         bootstrappingSize=bootstrappingSize,
                         bootstrappingSeed=bootstrappingSeed,
                         numMultiThreshold=numMultiThreshold)
    z = zOfChiSquare(alpha=alpha,
                     D_riser_number=D_riser_number,
                     gDimension=gEllipsoidalDimension,
                     numMultiThreshold=numMultiThreshold)

    radius = z/data.shape[0]
    return optimization(D_riser_number=D_riser_number, eta_lb=eta_lb, eta_ub=eta_ub, nu=nu,
                        threshold_level=threshold,
                        h=ObjectiveFunction,
                        g_Es=MomentConstraintFunctions,
                        mu_value=mu, Sigma=Sigma, radius=radius)


def OptimizationPlainKolmogorov(data: np.ndarray,
                                threshold: float,
                                ObjectiveFunction: PolynomialFunction,
                                bootstrappingSize: int,
                                bootstrappingSeed: int,
                                alpha: float,
                                numMultiThreshold: int) -> float:
    ## (0, ks)
    newObjectiveFunction = deepcopy(ObjectiveFunction)
    D_riser_number = 0
    dataOverThreshold = np.sort(data[data > threshold])
    sizeOverThreshold = np.sum(data > threshold)
    sizeOnData = data.shape[0]

    z = zOfKolmogorov(alpha=alpha,
                      D_riser_number=D_riser_number,
                      numMultiThreshold=numMultiThreshold)
    mu_lb_value = np.maximum(0, (sizeOnData+1-np.arange(
        sizeOnData-sizeOverThreshold+1, sizeOnData+1))/sizeOnData-z/np.sqrt(sizeOnData))
    mu_ub_value = np.minimum(1, (sizeOnData-np.arange(
        sizeOnData-sizeOverThreshold+1, sizeOnData+1))/sizeOnData+z/np.sqrt(sizeOnData))

    ConstraintFunctions = [PolynomialFunction(
        [xi, np.inf], [[0]*0+[1]]) for xi in dataOverThreshold]

    return optimization(D_riser_number=D_riser_number,
                        threshold_level=threshold,
                        h=newObjectiveFunction,
                        g_Rs=ConstraintFunctions,
                        mu_lb_value=mu_lb_value, mu_ub_value=mu_ub_value)


def OptimizationMonotoneKolmogorov(data: np.ndarray,
                                   threshold: float,
                                   ObjectiveFunction: PolynomialFunction,
                                   bootstrappingSize: int,
                                   bootstrappingSeed: int,
                                   alpha: float,
                                   numMultiThreshold: int) -> float:
    ## (1, ks)
    newObjectiveFunction = deepcopy(ObjectiveFunction)
    D_riser_number = 1
    dataOverThreshold = np.sort(data[data > threshold])
    sizeOverThreshold = np.sum(data > threshold)
    sizeOnData = data.shape[0]

    z = zOfKolmogorov(alpha=alpha,
                      D_riser_number=D_riser_number,
                      numMultiThreshold=numMultiThreshold)
    mu_lb_value = np.maximum(0, (sizeOnData+1-np.arange(
        sizeOnData-sizeOverThreshold+1, sizeOnData+1))/sizeOnData-z/np.sqrt(sizeOnData))
    mu_ub_value = np.minimum(1, (sizeOnData-np.arange(
        sizeOnData-sizeOverThreshold+1, sizeOnData+1))/sizeOnData+z/np.sqrt(sizeOnData))

    ConstraintFunctions = [PolynomialFunction(
        [xi, np.inf], [[0]*0+[1]]) for xi in dataOverThreshold]

    eta = etaSpecification(data=data,
                           threshold=threshold,
                           alpha=alpha,
                           D_riser_number=D_riser_number,
                           bootstrappingSize=bootstrappingSize,
                           bootstrappingSeed=bootstrappingSeed,
                           numMultiThreshold=numMultiThreshold)

    return optimization(D_riser_number=D_riser_number,
                        eta=eta,
                        threshold_level=threshold,
                        h=newObjectiveFunction,
                        g_Rs=ConstraintFunctions,
                        mu_lb_value=mu_lb_value, mu_ub_value=mu_ub_value)


def OptimizationConvexKolmogorov(data: np.ndarray,
                                 threshold: float,
                                 ObjectiveFunction: PolynomialFunction,
                                 bootstrappingSize: int,
                                 bootstrappingSeed: int,
                                 alpha: float,
                                 numMultiThreshold: int) -> float:
    ## (2, ks)
    newObjectiveFunction = deepcopy(ObjectiveFunction)
    D_riser_number = 2
    dataOverThreshold = np.sort(data[data > threshold])
    sizeOverThreshold = np.sum(data > threshold)
    sizeOnData = data.shape[0]

    z = zOfKolmogorov(alpha=alpha,
                      D_riser_number=D_riser_number,
                      numMultiThreshold=numMultiThreshold)
    mu_lb_value = np.maximum(0, (sizeOnData+1-np.arange(
        sizeOnData-sizeOverThreshold+1, sizeOnData+1))/sizeOnData-z/np.sqrt(sizeOnData))
    mu_ub_value = np.minimum(1, (sizeOnData-np.arange(
        sizeOnData-sizeOverThreshold+1, sizeOnData+1))/sizeOnData+z/np.sqrt(sizeOnData))

    ConstraintFunctions = [PolynomialFunction(
        [xi, np.inf], [[0]*0+[1]]) for xi in dataOverThreshold]

    [eta_lb, eta_ub] = etaSpecification(data=data,
                                        threshold=threshold,
                                        alpha=alpha,
                                        D_riser_number=D_riser_number,
                                        bootstrappingSize=bootstrappingSize,
                                        bootstrappingSeed=bootstrappingSeed,
                                        numMultiThreshold=numMultiThreshold)

    nu = nuSpecification(data=data,
                         threshold=threshold,
                         alpha=alpha,
                         bootstrappingSize=bootstrappingSize,
                         bootstrappingSeed=bootstrappingSeed,
                         numMultiThreshold=numMultiThreshold)

    return optimization(D_riser_number=D_riser_number,
                        eta_lb=eta_lb, eta_ub=eta_ub, nu=nu,
                        threshold_level=threshold,
                        h=newObjectiveFunction,
                        g_Rs=ConstraintFunctions,
                        mu_lb_value=mu_lb_value, mu_ub_value=mu_ub_value)


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

    if rightEndPointObjective == np.inf:
        h = PolynomialFunction(
            [leftEndPointObjective, np.inf], [[1]])
    else:
        h = PolynomialFunction(
            [leftEndPointObjective, rightEndPointObjective, np.inf], [[1], [0]])

    if isinstance(thresholdPercentage, float):
        numMultiThreshold = 1
        thresholds = [np.quantile(inputData, thresholdPercentage)]
    else:
        assert isinstance(thresholdPercentage, list) and len(
            thresholdPercentage) >= 2
        assert all(isinstance(
            eachThresholdPercentage, float) for eachThresholdPercentage in thresholdPercentage)
        numMultiThreshold = len(thresholdPercentage)
        thresholds = [np.quantile(inputData, eachThresholdPercentage)
                      for eachThresholdPercentage in thresholdPercentage]

    if D == 0:
        return np.min([OptimizationPlainKolmogorov(data=inputData,
                                                   threshold=threshold,
                                                   ObjectiveFunction=h,
                                                   bootstrappingSize=bootstrappingSize,
                                                   bootstrappingSeed=bootstrappingSeed,
                                                   alpha=alpha,
                                                   numMultiThreshold=numMultiThreshold) for threshold in thresholds])
    if D == 1:
        return np.min([OptimizationMonotoneKolmogorov(data=inputData,
                                                      threshold=threshold,
                                                      ObjectiveFunction=h,
                                                      bootstrappingSize=bootstrappingSize,
                                                      bootstrappingSeed=bootstrappingSeed,
                                                      alpha=alpha,
                                                      numMultiThreshold=numMultiThreshold) for threshold in thresholds])
    if D == 2:
        return np.min([OptimizationConvexKolmogorov(data=inputData,
                                                    threshold=threshold,
                                                    ObjectiveFunction=h,
                                                    bootstrappingSize=bootstrappingSize,
                                                    bootstrappingSeed=bootstrappingSeed,
                                                    alpha=alpha,
                                                    numMultiThreshold=numMultiThreshold) for threshold in thresholds])


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
    if rightEndPointObjective == np.inf:
        h = PolynomialFunction(
            [leftEndPointObjective, np.inf], [[1]])
    else:
        h = PolynomialFunction(
            [leftEndPointObjective, rightEndPointObjective, np.inf], [[1], [0]])

    if rightEndPointObjective == np.inf:
        h = PolynomialFunction(
            [leftEndPointObjective, np.inf], [[1]])
    else:
        h = PolynomialFunction(
            [leftEndPointObjective, rightEndPointObjective, np.inf], [[1], [0]])

    if isinstance(thresholdPercentage, float):
        numMultiThreshold = 1
        thresholds = [np.quantile(inputData, thresholdPercentage)]
        g_EsList = [[PolynomialFunction([thresholds[0], np.inf], [[0] * i + [1]])
                     for i in range(gEllipsoidalDimension)]]
        muList = [np.array([np.sum(inputData**power*(inputData > thresholds[0])) /
                            inputData.size for power in range(gEllipsoidalDimension)])]
        SigmaList = [np.cov(np.vstack(
            [(inputData > thresholds[0])*1.0*inputData**power for power in range(gEllipsoidalDimension)]))]
    else:
        assert isinstance(thresholdPercentage, list) and len(
            thresholdPercentage) >= 2
        assert all(isinstance(
            eachThresholdPercentage, float) for eachThresholdPercentage in thresholdPercentage)

        numMultiThreshold = len(thresholdPercentage)
        thresholds = [np.quantile(inputData, eachThresholdPercentage)
                      for eachThresholdPercentage in thresholdPercentage]
        g_EsList = [[PolynomialFunction([threshold, np.inf], [[0] * i + [1]])
                     for i in range(gEllipsoidalDimension)] for threshold in thresholds]
        muList = [np.array([np.sum(inputData**power*(inputData > threshold)) /
                            inputData.size for power in range(gEllipsoidalDimension)]) for threshold in thresholds]
        SigmaList = [np.cov(np.vstack(
            [(inputData > threshold)*1.0*inputData**power for power in range(gEllipsoidalDimension)])) for threshold in thresholds]

    if D == 0:
        return np.min([OptimizationPlainChiSquare(data=inputData,
                                                  threshold=threshold,
                                                  ObjectiveFunction=h,
                                                  MomentConstraintFunctions=g_Es,
                                                  mu=mu,
                                                  Sigma=Sigma,
                                                  bootstrappingSize=bootstrappingSize,
                                                  bootstrappingSeed=bootstrappingSeed,
                                                  alpha=alpha,
                                                  numMultiThreshold=numMultiThreshold) for threshold, g_Es, mu, Sigma in zip(thresholds, g_EsList, muList, SigmaList)])
    if D == 1:
        return np.min([OptimizationMonetoneChiSquare(data=inputData,
                                                     threshold=threshold,
                                                     ObjectiveFunction=h,
                                                     MomentConstraintFunctions=g_Es,
                                                     mu=mu,
                                                     Sigma=Sigma,
                                                     bootstrappingSize=bootstrappingSize,
                                                     bootstrappingSeed=bootstrappingSeed,
                                                     alpha=alpha,
                                                     numMultiThreshold=numMultiThreshold) for threshold, g_Es, mu, Sigma in zip(thresholds, g_EsList, muList, SigmaList)])
    if D == 2:
        return np.min([OptimizationConvexChiSquare(data=inputData,
                                                   threshold=threshold,
                                                   ObjectiveFunction=h,
                                                   MomentConstraintFunctions=g_Es,
                                                   mu=mu,
                                                   Sigma=Sigma,
                                                   bootstrappingSize=bootstrappingSize,
                                                   bootstrappingSeed=bootstrappingSeed,
                                                   alpha=alpha,
                                                   numMultiThreshold=numMultiThreshold) for threshold, g_Es, mu, Sigma in zip(thresholds, g_EsList, muList, SigmaList)])


if __name__ == '__main__':
    pass
