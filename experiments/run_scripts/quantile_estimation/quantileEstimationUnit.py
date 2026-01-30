import droevt.utils.synthetic_data_generator as data_utils
import droevt.routine as droevt_routine
import typing
from scipy.stats import gamma, lognorm, pareto, genpareto
import numpy as np

def quantileEstimationWithRectangularConstraintBinarySearchUnit(D: int, inputData: np.ndarray,
                                       thresholdPercentage: typing.Union[float, typing.List[float]],
                                       quantitleValue: float, alpha: float, random_state: int) -> float:
    assert D == 1 or D == 2
    
    if isinstance(thresholdPercentage, float):
        startQuantilePoint = np.quantile(inputData, thresholdPercentage)
    else:
        assert isinstance(thresholdPercentage, list) and len(
            thresholdPercentage) >= 2
        assert all(isinstance(
            eachThresholdPercentage, float) for eachThresholdPercentage in thresholdPercentage)
        startQuantilePoint = np.quantile(inputData, np.max(thresholdPercentage))

    targetValue = 1 - quantitleValue
    currentValue = np.inf
    lhsPoint = startQuantilePoint
    rhsPoint = np.inf
    midPoint = 2*lhsPoint
    ## we assume that max P(X>=startQuantilePoint) > targetValue.
    while (np.abs(currentValue-targetValue)/targetValue > 1e-6 and (rhsPoint-lhsPoint)/lhsPoint>1e-6):
        currentValue = droevt_routine.optimization_with_rectangular_constraint(D,
                                                                               inputData,
                                                                               thresholdPercentage,
                                                                               alpha,
                                                                               midPoint, np.inf,
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
    while (np.abs(currentValue-targetValue)/targetValue > 1e-6 and (rhsPoint-lhsPoint)/lhsPoint>1e-6):
        currentValue = droevt_routine.optimization_with_ellipsodial_constraint(D,
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


def quantileEstimationPerRep(dataModule,
                              quantitleValue: float,
                              dataSize: int, thresholdPercentage: typing.Union[float, typing.List[float]],
                              gEllipsoidalDimension: int,
                              alpha: float,
                              random_state: int) -> typing.List[float]:

    inputData = data_utils.generate_synthetic_data(
        dataModule, data_utils.DISTRIBUTION_DEFAULT_PARAMETERS[dataModule.name], dataSize, random_state)
    outputPerRep = []
    # outputPerRep.append(quantileEstimationWithRectangularConstraintBinarySearchUnit(1, inputData,
    #                                                                               thresholdPercentage,
    #                                                                               quantitleValue, alpha, 7*random_state+1))

    # outputPerRep.append(quantileEstimationWithRectangularConstraintBinarySearchUnit(2, inputData,
    #                                                                               thresholdPercentage,
    #                                                                               quantitleValue, alpha, 7*random_state+1))

    outputPerRep.append(quantileEstimationBinarySearchUnit(0, inputData,
                                                         thresholdPercentage,
                                                         quantitleValue, gEllipsoidalDimension, alpha, 7*random_state+1))

    outputPerRep.append(quantileEstimationBinarySearchUnit(1, inputData,
                                                         thresholdPercentage,
                                                         quantitleValue, gEllipsoidalDimension, alpha, 7*random_state+1))

    outputPerRep.append(quantileEstimationBinarySearchUnit(2, inputData,
                                                         thresholdPercentage,
                                                         quantitleValue, gEllipsoidalDimension, alpha, 7*random_state+1))
    return outputPerRep


if __name__ == '__main__':
    dataSize = 500
    quantitleValue = 0.99
    trueValue = data_utils.get_quantile(
        gamma, quantitleValue, data_utils.DISTRIBUTION_DEFAULT_PARAMETERS[gamma])
    thresholdPercentage = 0.7
    alpha = 0.05
    random_state = 20220222
    gEllipsoidalDimension = 3
    print("A small example on quantile estimation--single threshold.")
    print(quantileEstimationPerRep(
        gamma, quantitleValue, dataSize, thresholdPercentage, gEllipsoidalDimension, alpha, random_state))
    # print(trueValue)
    print("A small example on quantile estimation--multiple thresholds.")
    thresholdPercentage = [0.65, 0.7, 0.75, 0.8]
    print(quantileEstimationPerRep(
        gamma, quantitleValue, dataSize, thresholdPercentage, gEllipsoidalDimension, alpha, random_state))
