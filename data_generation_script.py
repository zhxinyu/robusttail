from table5_central_import import *

def runner():
    randomSeed = 20220222
    stringToDataModule = {"gamma": gamma,
                          "lognorm": lognorm,
                          "pareto": pareto}
    metaDataDict = {"dataSize": 500}
    dataSources = [ "gamma", "lognorm", "pareto" ]
    
    nrep = 200
    for dataSource in dataSources:
        dataModule = stringToDataModule[dataSource]
        for nnrep in range(nrep):
            metaDataDict['random_state'] = randomSeed+nnrep
            inputData = dpu.RawDataGeneration(dataModule, 
                                              dpu.dataModuleToDefaultParamDict[dataModule], 
                                              metaDataDict['dataSize'], 
                                              metaDataDict['random_state'])
            np.savetxt(f"large_data/{dataSource}/default/randomseed={randomSeed+nnrep}.csv", inputData, delimiter=",")
                

if __name__ == '__main__':
    runner()