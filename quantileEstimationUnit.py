import dataPreparationUtils as dpu
import optimizationUnit as ou
import typing
from scipy.stats import gamma, lognorm, pareto, genpareto
import numpy as np


def quantileEstimationBinarySearchUnit(D: int, inputData: np.ndarray,
                                       thresholdPercentage: typing.Union[float, typing.List[float]],
                                       quantitleValue: float, gEllipsoidalDimension: int, alpha: float, random_state: int) -> float:
    if isinstance(thresholdPercentage, float):
        startQuantilePoint = np.quantile(inputData, thresholdPercentage)
    else:
        assert isinstance(thresholdPercentage, list) and len(
            thresholdPercentage) >= 2
        assert all(isinstance(
            eachThresholdPercentage, float) for eachThresholdPercentage in thresholdPercentage)
        startQuantilePoint = np.quantile(
            inputData, np.max(thresholdPercentage))

    targetValue = 1 - quantitleValue
    currentValue = np.inf
    lhsPoint = startQuantilePoint
    rhsPoint = np.inf
    midPoint = 2*lhsPoint
    ## we assume that max P(X>=startQuantilePoint) > targetValue.
    while np.abs(currentValue-targetValue) > 1e-5:
        currentValue = ou.OptimizationWithEllipsodialConstraint(D,
                                                                inputData,
                                                                thresholdPercentage,
                                                                alpha,
                                                                midPoint, np.inf, gEllipsoidalDimension,
                                                                inputData.size, 7*random_state+1)
        outputMidPoint = midPoint
        if currentValue > targetValue:
            lhsPoint = midPoint
        else:
            rhsPoint = midPoint
        if rhsPoint == np.inf:
            midPoint = 2*lhsPoint
        else:
            midPoint = lhsPoint + (rhsPoint-lhsPoint)/2
    return outputMidPoint


def quantileEstimationnPerRep(dataModule,
                              quantitleValue: float,
                              dataSize: int, thresholdPercentage: typing.Union[float, typing.List[float]],
                              gEllipsoidalDimension: int,
                              alpha: float,
                              random_state: int) -> typing.List[float]:

    inputData = dpu.RawDataGeneration(
        dataModule, dpu.dataModuleToDefaultParamDict[dataModule], dataSize, random_state)
    outputPerRep = [0]*3
    outputPerRep[0] = quantileEstimationBinarySearchUnit(0, inputData,
                                                         thresholdPercentage,
                                                         quantitleValue, gEllipsoidalDimension, alpha, 7*random_state+1)

    outputPerRep[1] = quantileEstimationBinarySearchUnit(1, inputData,
                                                         thresholdPercentage,
                                                         quantitleValue, gEllipsoidalDimension, alpha, 7*random_state+1)

    outputPerRep[2] = quantileEstimationBinarySearchUnit(2, inputData,
                                                         thresholdPercentage,
                                                         quantitleValue, gEllipsoidalDimension, alpha, 7*random_state+1)
    return outputPerRep


if __name__ == '__main__':
    dataSize = 500
    quantitleValue = 0.99
    trueValue = dpu.endPointGeneration(
        gamma, quantitleValue, dpu.dataModuleToDefaultParamDict[gamma])
    thresholdPercentage = 0.7
    alpha = 0.05
    random_state = 20220222
    gEllipsoidalDimension = 3
    print("A small example on quantile estimation--single threshold.")
    print(quantileEstimationnPerRep(
        gamma, quantitleValue, dataSize, thresholdPercentage, gEllipsoidalDimension, alpha, random_state))
    # print(trueValue)
    print("A small example on quantile estimation--multiple thresholds.")
    thresholdPercentage = [0.65, 0.7, 0.75, 0.8]
    print(quantileEstimationnPerRep(
        gamma, quantitleValue, dataSize, thresholdPercentage, gEllipsoidalDimension, alpha, random_state))
