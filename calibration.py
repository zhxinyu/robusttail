from scipy.stats import norm, chi2, kstwobign
import numpy as np
import random
import typing

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
numpy2ri.activate()
from rpy2.robjects.packages import importr
importr('base')
importr('utils')
importr('stats')
importr('ks')

############################################################################################################
############################################################################################################
#### Helper Function
############################################################################################################
############################################################################################################
def etaGeneration(data: np.ndarray, 
                  pointEstimate: float, 
                  bootstrappingFlag: bool, 
                  bootstrappingSize: int, 
                  bootstrappingSeed) -> typing.Union[float, typing.List[float]]:
    # Generate eta based on obseravtions.
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
            boostrappingData = random.choices(data, k = bootstrappingSize)
            outputList.append(ro.r(
                '''
                data = c({:})
                (kdde(x = data, deriv.order = 0, eval.points = {:}))$estimate
                '''.format(','.join([str(eachData)for eachData in boostrappingData]), pointEstimate)
                )[0]
             )
        return outputList
    
def nuGeneration(data: np.ndarray, 
                 pointEstimate: float, 
                 bootstrappingFlag: bool, 
                 bootstrappingSize: int, 
                 bootstrappingSeed: int) -> typing.Union[float, typing.List[float]]:
    # Generate nu based on obseravtions.    
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
            boostrappingData = random.choices(data, k = bootstrappingSize)
            outputList.append(ro.r(
                '''
                data = c({:})
                (kdde(x = data, deriv.order = 1, eval.points = {:}))$estimate
                '''.format(','.join([str(eachData)for eachData in boostrappingData]), pointEstimate)
                )[0]
             )
        return outputList

############################################################################################################
############################################################################################################
#### Ellipsoidal Specification (chi square)
############################################################################################################
############################################################################################################
def etaEllipsoidalSpecification(data: np.ndarray, 
                                threshold: float, 
                                bootstrappingSize: int, 
                                bootstrappingSeed, alpha: float, 
                                D_riser_number: int, 
                                numMultiThreshold: int):
    # Generate eta under ellipsoidal specification (chi square)
    etaBoostrapping = etaGeneration(data, 
                                    threshold, 
                                    bootstrappingFlag = True, 
                                    bootstrappingSize = bootstrappingSize, 
                                    bootstrappingSeed = bootstrappingSeed) 
    assert D_riser_number == 1 or D_riser_number == 2 
    if D_riser_number == 1:
        quantile = 1-alpha/(numMultiThreshold + 1)
        return np.quantile(a = etaBoostrapping, q = quantile)        
    else:        
        return np.quantile(a = etaBoostrapping, q = [alpha/(4*numMultiThreshold + 2),1-alpha/(4*numMultiThreshold + 2)])

def nuEllipsoidalSpecification(data: np.ndarray, 
                            threshold: float, 
                            bootstrappingSize: int, 
                            bootstrappingSeed: int, 
                            alpha: float, 
                            numMultiThreshold: int):
    # Generate nu under ellipsoidal specification (chi square)
    quantile = alpha/(2*numMultiThreshold+1)
    etaBoostrapping = nuGeneration(data, 
                                   threshold,
                                   bootstrappingFlag = True,
                                   bootstrappingSize = bootstrappingSize, 
                                   bootstrappingSeed = bootstrappingSeed)
    return -np.quantile(a = etaBoostrapping, q = quantile)        
    

def zOfChiSquare(alpha: float, 
                 D_riser_number: int, 
                 gDimension: int, 
                 numMultiThreshold: int):
    # Generate z value under chi square distribution
    assert D_riser_number == 1 or D_riser_number == 2 or D_riser_number == 0
    if D_riser_number == 1:
        quantile = 1-alpha/(numMultiThreshold + 1)
        return chi2.ppf(q = quantile,df = gDimension)
    elif D_riser_number == 2:        
        quantile = 1-alpha/(2*numMultiThreshold + 1)
        return chi2.ppf(q = quantile,df = gDimension)
    else: ## D_riser_number == 0
        quantile = 1-alpha/numMultiThreshold
        return chi2.ppf(q = quantile,df = gDimension)        

############################################################################################################
############################################################################################################
#### Rectangular Specification (Kolmogorov–Smirnov test)
############################################################################################################
############################################################################################################
def zOfKolmogorov(alpha: float, 
                  D_riser_number: int, 
                  sizeOverThreshold: int, 
                  numMultiThreshold: int):
    # Generate z value under Kolmogorov distribution    
    assert D_riser_number == 1 or D_riser_number == 2 or D_riser_number == 0
    if D_riser_number == 1:
        quantile = 1-alpha/(2*numMultiThreshold + 1)
        return kstwobign.ppf(q = quantile)/np.sqrt(sizeOverThreshold)
    elif D_riser_number == 2:
        quantile = 1-alpha/(3*numMultiThreshold + 1)
        return kstwobign.ppf(q = quantile)/np.sqrt(sizeOverThreshold)
    else:        
        quantile = 1-alpha/(numMultiThreshold + 1)
        return kstwobign.ppf(q = quantile)/np.sqrt(sizeOverThreshold)

def pHatRectangularSpecification(sizeOnData: float, 
                                sizeOverThreshold: float,
                                sigmaHat: float,                                  
                                alpha: float, 
                                D_riser_number: int,
                                numMultiThreshold: int):
    assert D_riser_number == 1 or D_riser_number == 2 or D_riser_number == 0
    if D_riser_number == 1:
        quantile = 1-alpha/(2*numMultiThreshold+1)
        return sizeOverThreshold/sizeOnData+norm.ppf(q = quantile)*sigmaHat/np.sqrt(sizeOnData)
    elif D_riser_number == 2:
        quantile = 1-alpha/(3*numMultiThreshold+1)
        return sizeOverThreshold/sizeOnData+norm.ppf(q = quantile)*sigmaHat/np.sqrt(sizeOnData)    
    else:
        quantile = 1-alpha/(numMultiThreshold+1)
        return sizeOverThreshold/sizeOnData+norm.ppf(q = quantile)*sigmaHat/np.sqrt(sizeOnData)    
        
def etaRectangularSpecification(dataOverThreshold: np.ndarray, 
                                threshold: float, 
                                alpha: float, 
                                D_riser_number: int,
                                bootstrappingSize: int, 
                                bootstrappingSeed: float,  
                                numMultiThreshold: int):
    assert D_riser_number == 1 or D_riser_number == 2
    etaBoostrapping = etaGeneration(dataOverThreshold, 
                                    threshold, 
                                    bootstrappingFlag = True, 
                                    bootstrappingSize = bootstrappingSize, 
                                    bootstrappingSeed = bootstrappingSeed)
    if D_riser_number == 1:
        quantile = 1-alpha/(2*numMultiThreshold + 1)
        return np.quantile(a = etaBoostrapping, q = quantile)        
    else:
        return np.quantile(a = etaBoostrapping, q = [alpha/(6*numMultiThreshold + 2),1-alpha/(6*numMultiThreshold + 2)])
    
def nuRectangularSpecification(dataOverThreshold: np.ndarray, 
                                threshold: float, 
                                bootstrappingSize: int, 
                                bootstrappingSeed: int, 
                                alpha: float, 
                                numMultiThreshold: int):
    etaBoostrapping = nuGeneration(dataOverThreshold, 
                                   threshold, 
                                   bootstrappingFlag = True, 
                                   bootstrappingSize = bootstrappingSize, 
                                   bootstrappingSeed = bootstrappingSeed)    
    quantile = alpha/(3*numMultiThreshold+1)
    return -np.quantile(a = etaBoostrapping, q = quantile)
                
if __name__ == '__main__':
    test_PolynomialFunction_integration_riser()