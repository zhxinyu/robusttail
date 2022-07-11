from rpy2.robjects.packages import importr
from rpy2.rinterface_lib.embedded import RRuntimeError
from scipy.stats import norm, chi2, kstwobign
import numpy as np
import random
import typing

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
numpy2ri.activate()
importr('base')
utils = importr('utils')
importr('stats')
try:
    importr('ks')
except RRuntimeError:
    utils.install_packages('ks', contribulr="https://cran.microsoft.com/")
    importr('ks')


############################################################################################################
############################################################################################################
# Helper Function to generate parameters in optimization problems.
############################################################################################################
############################################################################################################


def etaGeneration(data: np.ndarray,
                  pointEstimate: float,
                  bootstrappingFlag: bool,
                  bootstrappingSize: int,
                  bootstrappingSeed: int) -> typing.Union[float, typing.List[float]]:
    # Generate eta based on observations.
    if not bootstrappingFlag:
        return ro.r(
            '''
        data = c({:})
        (kdde(x = data, deriv.order = 0, eval.points = {:}))$estimate
        '''.format(','.join([str(eachData)for eachData in data.tolist()]), pointEstimate)
        )[0]
    else:
        random.seed(bootstrappingSeed)
        outputList = []
        for _ in range(bootstrappingSize):
            boostrappingData = random.choices(data, k=bootstrappingSize)
            outputList.append(ro.r(
                '''
                data = c({:})
                (kdde(x = data, deriv.order = 0, eval.points = {:}))$estimate
                '''.format(','.join([str(eachData) for eachData in boostrappingData]), pointEstimate)
            )[0])
        return outputList


def etaSpecification(data: np.ndarray,
                     threshold: float,
                     alpha: float,
                     bootstrappingSize: int,
                     bootstrappingSeed,
                     D_riser_number: int,
                     numMultiThreshold: int):
    etaBoostrapping = etaGeneration(data,
                                    threshold,
                                    bootstrappingFlag=True,
                                    bootstrappingSize=bootstrappingSize,
                                    bootstrappingSeed=bootstrappingSeed)
    assert D_riser_number == 1 or D_riser_number == 2
    if D_riser_number == 1:
        quantile = 1-alpha/(numMultiThreshold + 1)
        return np.quantile(a=etaBoostrapping, q=quantile)
    else:
        return np.quantile(a=etaBoostrapping, q=[alpha/(4*numMultiThreshold + 2), 1-alpha/(4*numMultiThreshold + 2)])


def nuGeneration(data: np.ndarray,
                 pointEstimate: float,
                 bootstrappingFlag: bool,
                 bootstrappingSize: int,
                 bootstrappingSeed: int) -> typing.Union[float, typing.List[float]]:
    # Generate nu based on observations.
    if not bootstrappingFlag:
        return ro.r(
            '''
                data = c({:})
                (kdde(x = data, deriv.order = 1, eval.points = {:}))$estimate
                '''.format(','.join([str(eachData)for eachData in data.tolist()]), pointEstimate)
        )[0]
    else:
        random.seed(bootstrappingSeed)
        outputList = []
        for _ in range(bootstrappingSize):
            boostrappingData = random.choices(data, k=bootstrappingSize)
            outputList.append(ro.r(
                '''
                data = c({:})
                (kdde(x = data, deriv.order = 1, eval.points = {:}))$estimate
                '''.format(','.join([str(eachData) for eachData in boostrappingData]), pointEstimate)
            )[0])
        return outputList


def nuSpecification(data: np.ndarray,
                    threshold: float,
                    alpha: float,
                    bootstrappingSize: int,
                    bootstrappingSeed: int,
                    numMultiThreshold: int):
    quantile = alpha/(2*numMultiThreshold+1)
    etaBoostrapping = nuGeneration(data,
                                   threshold,
                                   bootstrappingFlag=True,
                                   bootstrappingSize=bootstrappingSize,
                                   bootstrappingSeed=bootstrappingSeed)
    return -np.quantile(a=etaBoostrapping, q=quantile)


############################################################################################################
############################################################################################################
# Ellipsoidal Specification (chi square)
############################################################################################################
############################################################################################################

def zOfChiSquare(alpha: float,
                 D_riser_number: int,
                 gDimension: int,
                 numMultiThreshold: int):
    # Generate z value under chi square distribution
    assert D_riser_number == 1 or D_riser_number == 2 or D_riser_number == 0
    if D_riser_number == 1:
        quantile = 1-alpha/(numMultiThreshold + 1)
        return chi2.ppf(q=quantile, df=gDimension)
    elif D_riser_number == 2:
        quantile = 1-alpha/(2*numMultiThreshold + 1)
        return chi2.ppf(q=quantile, df=gDimension)
    else:  # D_riser_number == 0
        quantile = 1-alpha/numMultiThreshold
        return chi2.ppf(q=quantile, df=gDimension)

############################################################################################################
############################################################################################################
# Rectangular Specification (Kolmogorov–Smirnov test)
############################################################################################################
############################################################################################################


def zOfKolmogorov(alpha: float,
                  D_riser_number: int,
                  numMultiThreshold: int):
    # Generate z value under Kolmogorov distribution
    assert D_riser_number == 1 or D_riser_number == 2 or D_riser_number == 0
    if D_riser_number == 1:
        quantile = 1-alpha/(numMultiThreshold + 1)
        return kstwobign.ppf(q=quantile)
    elif D_riser_number == 2:
        quantile = 1-alpha/(2*numMultiThreshold + 1)
        return kstwobign.ppf(q=quantile)
    else:
        quantile = 1-alpha/(numMultiThreshold)
        return kstwobign.ppf(q=quantile)


if __name__ == '__main__':
    pass
