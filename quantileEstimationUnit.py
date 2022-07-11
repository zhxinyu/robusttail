import dataPreparationUtils as dpu
import optimizationUnit as ou
import typing
from scipy.stats import gamma, lognorm, pareto, genpareto
import numpy as np


def quantileEstimationBinarySearchUnit(D: int, inputData: np.ndarray,
                                       thresholdPercentage: typing.Union[float, typing.List[float]],
                                       quantitleValue: float, alpha: float, random_state: int, 
                                       constraintType: str, 
                                       gEllipsoidalDimension: int) -> float:
    if constraintType != "RectangularConstraint" and constraintType != "EllipsodialConstraint":
        assert False
    if type(thresholdPercentage) == float:
        startQuantilePoint = np.quantile(inputData, thresholdPercentage)
    else:
        startQuantilePoint = np.max([np.quantile(inputData, eachThresholdPercentage)
                                     for eachThresholdPercentage in thresholdPercentage])
    targetValue = 1 - quantitleValue
    currentValue = np.inf
    lhsPoint = startQuantilePoint
    rhsPoint = np.inf
    midPoint = 2*lhsPoint
    ## we assume that max P(X>=startQuantilePoint) > targetValue.
    while np.abs(currentValue-targetValue) > 1e-6:
        if constraintType == "RectangularConstraint":
            currentValue = ou.OptimizationWithRectangularConstraint(D,
                                inputData,
                                thresholdPercentage,
                                alpha,
                                midPoint, np.inf,
                                inputData.size, 7*random_state+1)            
        else:
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
        print(currentValue, lhsPoint)
    return outputMidPoint


def quantileEstimationWithRectangularConstraintPerRep(dataModule,
                                                      quantitleValue: float,
                                                      dataSize: int, thresholdPercentage: typing.Union[float, typing.List[float]],
                                                      alpha: float,
                                                      random_state: int) -> typing.List[float]:
    inputData = dpu.RawDataGeneration(
        dataModule, dpu.dataModuleToDefaultParamDict[dataModule], dataSize, random_state)
    outputPerRep = [0]*2
    outputPerRep[1] = quantileEstimationBinarySearchUnit(1, inputData,
                                                         thresholdPercentage,
                                                         quantitleValue, alpha, 7*random_state+1, "RectangularConstraint", 0)
    outputPerRep[2] = quantileEstimationBinarySearchUnit(2, inputData,
                                                         thresholdPercentage,
                                                         quantitleValue, alpha, 7*random_state+1, "RectangularConstraint", 0)
    return outputPerRep


def quantileEstimationWithEllipsodialConstraintPerRep(dataModule,
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
                                                         quantitleValue, alpha, 7*random_state+1, "EllipsodialConstraint", gEllipsoidalDimension)

    outputPerRep[1] = quantileEstimationBinarySearchUnit(1, inputData,
                                                         thresholdPercentage,
                                                         quantitleValue, alpha, 7*random_state+1, "EllipsodialConstraint", gEllipsoidalDimension)

    outputPerRep[2] = quantileEstimationBinarySearchUnit(2, inputData,
                                                         thresholdPercentage,
                                                         quantitleValue, alpha, 7*random_state+1, "EllipsodialConstraint", gEllipsoidalDimension)
    return outputPerRep


def quantileEstimationnPerRep(dataModule,
                              quantitleValue: float,
                              dataSize: int, thresholdPercentage: typing.Union[float, typing.List[float]],
                              gEllipsoidalDimension: int,
                              alpha: float,
                              random_state: int) -> typing.List[float]:
    outputPerRep = []
    # quantileEstimationWithRectangularConstraintPerRep(dataModule,
    #                                                                  quantitleValue,
    #                                                                  dataSize, thresholdPercentage,
    #                                                                  alpha,
    #                                                                  random_state)
    outputPerRep.extend(
        quantileEstimationWithEllipsodialConstraintPerRep(dataModule,
                                                          quantitleValue,
                                                          dataSize, thresholdPercentage,
                                                          gEllipsoidalDimension,
                                                          alpha,
                                                          random_state))
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
    print(quantileEstimationnPerRep(
        gamma, quantitleValue, dataSize, thresholdPercentage, gEllipsoidalDimension, alpha, random_state))
    print(trueValue)
    thresholdPercentage = [0.65, 0.7, 0.75, 0.8]
    print(quantileEstimationnPerRep(
        gamma, quantitleValue, dataSize, thresholdPercentage, gEllipsoidalDimension, alpha, random_state))
