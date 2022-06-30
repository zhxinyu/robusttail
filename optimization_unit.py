from optimization_engine import optimization, PolynomialFunction
from calibration import etaEllipsoidalSpecification, nuEllipsoidalSpecification, zOfChiSquare, zOfKolmogorov, pHatRectangularSpecification, etaRectangularSpecification, nuRectangularSpecification
import numpy as np
from copy import deepcopy
from typing import List
def Optimization_Plain_ChiSquare(data: np.ndarray, 
                                 threshold: float, 
                                 ObjectiveFunction: PolynomialFunction,
                                 MomentConstraintFunctions: List[PolynomialFunction],
                                 mu : np.ndarray, 
                                 Sigma: np.ndarray,
                                 bootstrappingSize: int,
                                 bootstrappingSeed: int,
                                 alpha: float,
                                 numMultiThreshold: int)-> float: 
    ## (0, chi2)        
    D_riser_number = 0
    gEllipsoidalDimension = len(MomentConstraintFunctions)
    z = zOfChiSquare(alpha = alpha,
                     D_riser_number = D_riser_number,
                     gDimension = gEllipsoidalDimension,
                     numMultiThreshold = numMultiThreshold)
    radius = z/data.shape[0]
    return optimization(D_riser_number = D_riser_number,
                        threshold_level = threshold,
                        h = ObjectiveFunction,
                        g_Es = MomentConstraintFunctions,
                        mu_value = mu, Sigma = Sigma, radius = radius)


def Optimization_Monetone_ChiSquare(data: np.ndarray, 
                              threshold: float, 
                              ObjectiveFunction: PolynomialFunction,
                              MomentConstraintFunctions: List[PolynomialFunction],
                              mu : np.ndarray, 
                              Sigma: np.ndarray,
                              bootstrappingSize: int,
                              bootstrappingSeed: int,
                              alpha: float,
                              numMultiThreshold: int)-> float: 
    ## (1, chi2)    
    D_riser_number = 1
    gEllipsoidalDimension = len(MomentConstraintFunctions)
    eta = etaEllipsoidalSpecification(data = data,
                                      threshold = threshold,
                                      bootstrappingSize = bootstrappingSize,
                                      bootstrappingSeed = bootstrappingSeed,
                                      alpha = alpha,
                                      D_riser_number = D_riser_number,
                                      numMultiThreshold = numMultiThreshold)
    z = zOfChiSquare(alpha = alpha,
                     D_riser_number = D_riser_number,
                     gDimension = gEllipsoidalDimension,
                     numMultiThreshold = numMultiThreshold)
    radius = z/data.shape[0]
    return optimization(D_riser_number = D_riser_number, 
                        eta = eta,
                        threshold_level = threshold,
                        h = ObjectiveFunction,
                        g_Es = MomentConstraintFunctions,
                        mu_value = mu, Sigma = Sigma, radius = radius)
                                      
def Optimization_Convex_ChiSquare(data: np.ndarray, 
                            threshold: float, 
                            ObjectiveFunction: PolynomialFunction,
                            MomentConstraintFunctions: List[PolynomialFunction],
                            mu : np.ndarray, 
                            Sigma: np.ndarray,
                            bootstrappingSize: int,
                            bootstrappingSeed: int,
                            alpha: float,
                            numMultiThreshold: int)-> float: 
    ## (2, chi2)    
    D_riser_number = 2    
    gEllipsoidalDimension = len(MomentConstraintFunctions)

    [eta_lb, eta_ub] = etaEllipsoidalSpecification(data = data,
                                                   threshold = threshold,
                                                   bootstrappingSize = bootstrappingSize,
                                                   bootstrappingSeed = bootstrappingSeed,
                                                   alpha = alpha,
                                                   D_riser_number = D_riser_number,
                                                   numMultiThreshold = numMultiThreshold)
    
    nu = nuEllipsoidalSpecification(data = data, 
                                    threshold = threshold,
                                    bootstrappingSize = bootstrappingSize,
                                    bootstrappingSeed = bootstrappingSeed,
                                    alpha = alpha,
                                    numMultiThreshold = numMultiThreshold)
    z = zOfChiSquare(alpha = alpha,
                     D_riser_number = D_riser_number,
                     gDimension = gEllipsoidalDimension,
                     numMultiThreshold = numMultiThreshold)
    
    radius = z/data.shape[0]
    return optimization(D_riser_number = D_riser_number, eta_lb = eta_lb, eta_ub = eta_ub, nu = nu,
                        threshold_level = threshold,
                        h = ObjectiveFunction,
                        g_Es = MomentConstraintFunctions,
                        mu_value = mu, Sigma = Sigma, radius = radius)      

def Optimization_Plain_Kolmogorov(data: np.ndarray, 
                                  threshold: float, 
                                  ObjectiveFunction: PolynomialFunction,
                                  bootstrappingSize: int,
                                  bootstrappingSeed: int,
                                  alpha: float,
                                  numMultiThreshold: int)-> float:
    ## (0, ks)
    newObjectiveFunction = deepcopy(ObjectiveFunction)
    D_riser_number = 0
    dataOverThreshold = np.sort(data[data>threshold])
    sizeOverThreshold = np.sum(data>threshold)
    sizeOnData = data.shape[0]
    sigmaHat = np.std(data>threshold)
    
    z = zOfKolmogorov(alpha = alpha, 
                      D_riser_number = D_riser_number, 
                      sizeOverThreshold = sizeOverThreshold, 
                      numMultiThreshold = numMultiThreshold)
    mu_lb_value = np.maximum(0,(sizeOverThreshold+1-np.arange(1, sizeOverThreshold+1))/sizeOverThreshold-z/np.sqrt(sizeOverThreshold))
    mu_ub_value = np.minimum(1,(sizeOverThreshold-np.arange(1, sizeOverThreshold+1))/sizeOverThreshold+z/np.sqrt(sizeOverThreshold))


    pHat = pHatRectangularSpecification(sizeOnData = sizeOnData, 
                                        sizeOverThreshold = sizeOverThreshold,
                                        sigmaHat = sigmaHat,
                                        alpha = alpha, 
                                        D_riser_number = D_riser_number,
                                        numMultiThreshold = numMultiThreshold)
    newObjectiveFunction.multiply(pHat)
    ConstraintFunctions = [PolynomialFunction([xi,np.inf],[[0]*0+[1]]) for xi in dataOverThreshold]
        
    return optimization(D_riser_number = D_riser_number, 
                        threshold_level=threshold,
                        h = newObjectiveFunction,
                        g_Rs = ConstraintFunctions, 
                        mu_lb_value = mu_lb_value, mu_ub_value = mu_ub_value)

    
def Optimization_Monotone_Kolmogorov(data: np.ndarray, 
                                     threshold: float, 
                                     ObjectiveFunction: PolynomialFunction,
                                     bootstrappingSize: int,
                                     bootstrappingSeed: int,
                                     alpha: float,
                                     numMultiThreshold: int)-> float: 
    ## (1, ks)
    newObjectiveFunction =  deepcopy(ObjectiveFunction)
    D_riser_number = 1
    dataOverThreshold = np.sort(data[data>threshold])
    sizeOverThreshold = np.sum(data>threshold)
    sizeOnData = data.shape[0]
    sigmaHat = np.std(data>threshold)
    
    z = zOfKolmogorov(alpha = alpha, 
                      D_riser_number = D_riser_number, 
                      sizeOverThreshold = sizeOverThreshold, 
                      numMultiThreshold = numMultiThreshold)
    mu_lb_value = np.maximum(0,(sizeOverThreshold+1-np.arange(1, sizeOverThreshold+1))/sizeOverThreshold-z/np.sqrt(sizeOverThreshold))
    mu_ub_value = np.minimum(1,(sizeOverThreshold-np.arange(1, sizeOverThreshold+1))/sizeOverThreshold+z/np.sqrt(sizeOverThreshold))


    pHat = pHatRectangularSpecification(sizeOnData = sizeOnData, 
                                        sizeOverThreshold = sizeOverThreshold,
                                        sigmaHat = sigmaHat,
                                        alpha = alpha, 
                                        D_riser_number = D_riser_number,
                                        numMultiThreshold = numMultiThreshold)
    newObjectiveFunction.multiply(pHat)
    ConstraintFunctions = [PolynomialFunction([xi,np.inf],[[0]*0+[1]]) for xi in dataOverThreshold]
    
    eta = etaRectangularSpecification(dataOverThreshold = dataOverThreshold, 
                                      threshold = threshold, 
                                      alpha = alpha, 
                                      D_riser_number = D_riser_number,
                                      bootstrappingSize = bootstrappingSize, 
                                      bootstrappingSeed = bootstrappingSeed,  
                                      numMultiThreshold = numMultiThreshold)
    
    return optimization(D_riser_number = D_riser_number, 
                        eta = eta,
                        threshold_level = threshold,
                        h = newObjectiveFunction,
                        g_Rs = ConstraintFunctions,
                        mu_lb_value = mu_lb_value, mu_ub_value = mu_ub_value)
                                      
def Optimization_Convex_Kolmogorov(data: np.ndarray, 
                                   threshold: float, 
                                   ObjectiveFunction: PolynomialFunction,
                                   bootstrappingSize: int,
                                   bootstrappingSeed: int,
                                   alpha: float,
                                   numMultiThreshold: int)-> float: 
    ## (2, ks)
    newObjectiveFunction =  deepcopy(ObjectiveFunction)
    D_riser_number = 2
    dataOverThreshold = np.sort(data[data>threshold])
    sizeOverThreshold = np.sum(data>threshold)
    sizeOnData = data.shape[0]
    sigmaHat = np.std(data>threshold)
    
    z = zOfKolmogorov(alpha = alpha, 
                      D_riser_number = D_riser_number, 
                      sizeOverThreshold = sizeOverThreshold, 
                      numMultiThreshold = numMultiThreshold)
    mu_lb_value = np.maximum(0,(sizeOverThreshold+1-np.arange(1, sizeOverThreshold+1))/sizeOverThreshold-z/np.sqrt(sizeOverThreshold))
    mu_ub_value = np.minimum(1,(sizeOverThreshold-np.arange(1, sizeOverThreshold+1))/sizeOverThreshold+z/np.sqrt(sizeOverThreshold))


    pHat = pHatRectangularSpecification(sizeOnData = sizeOnData, 
                                        sizeOverThreshold = sizeOverThreshold,
                                        sigmaHat = sigmaHat,
                                        alpha = alpha, 
                                        D_riser_number = D_riser_number,
                                        numMultiThreshold = numMultiThreshold)
    newObjectiveFunction.multiply(pHat)
    ConstraintFunctions = [PolynomialFunction([xi,np.inf],[[0]*0+[1]]) for xi in dataOverThreshold]
    
    [eta_lb, eta_ub] = etaRectangularSpecification(dataOverThreshold = dataOverThreshold, 
                                      threshold = threshold, 
                                      alpha = alpha, 
                                      D_riser_number = D_riser_number,
                                      bootstrappingSize = bootstrappingSize, 
                                      bootstrappingSeed = bootstrappingSeed,  
                                      numMultiThreshold = numMultiThreshold)
    
    nu = nuRectangularSpecification(dataOverThreshold = dataOverThreshold, 
                                    threshold = threshold, 
                                    bootstrappingSize = bootstrappingSize, 
                                    bootstrappingSeed = bootstrappingSeed, 
                                    alpha = alpha, 
                                    numMultiThreshold = numMultiThreshold)

    return optimization(D_riser_number = D_riser_number, 
                        eta_lb = eta_lb, eta_ub = eta_ub, nu = nu,
                        threshold_level=threshold,
                        h = newObjectiveFunction,
                        g_Rs = ConstraintFunctions,
                        mu_lb_value = mu_lb_value, mu_ub_value = mu_ub_value)
                                      
if __name__ == '__main__':
    pass