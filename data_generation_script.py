from table5_central_import import *

def runner():
    randomSeed = 20220222
    data_size=30
    stringToDataModule = {"gamma": gamma,
                          "lognorm": lognorm,
                          "pareto": pareto}
    metaDataDict = {"dataSize": data_size}
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
            file_dir = f"./n{data_size}/{dataSource}/default"
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            np.savetxt(f"{file_dir}/randomseed={randomSeed+nnrep}.csv", inputData, delimiter=",")
                

if __name__ == '__main__':
    runner()